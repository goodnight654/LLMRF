r"""
并行仿真加速模块
提供CPU多进程并行仿真功能
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# 设置环境变量
os.environ['HPEESOF_DIR'] = r"C:/Program Files/Keysight/ADS2025_Update1"

from keysight.ads import de
from keysight.ads import dataset


def simulate_single_design(args: Tuple[Dict, str, str]) -> Dict[str, Any]:
    """
    单个设计的仿真（可被多进程调用）
    
    Args:
        args: (design, workspace_base, library_name) 元组
        
    Returns:
        仿真结果字典或错误信息
    """
    design, workspace_base, library_name = args
    design_id = design['id']
    worker_id = os.getpid()
    
    # 每个worker使用独立的workspace路径
    workspace_path = f"{workspace_base}_worker_{worker_id}"
    output_base = Path(workspace_base).parent / "simulations"
    
    try:
        # 创建独立workspace和库
        if de.workspace_is_open():
            de.close_workspace()
        
        # 清理或创建workspace
        if os.path.exists(workspace_path):
            lib_defs = Path(workspace_path) / "lib.defs"
            if not lib_defs.exists():
                import shutil
                try:
                    shutil.rmtree(workspace_path)
                except Exception:
                    pass
                workspace = de.create_workspace(workspace_path)
                workspace.open()
            else:
                workspace = de.open_workspace(workspace_path)
        else:
            workspace = de.create_workspace(workspace_path)
            workspace.open()
        
        # 创建或打开库
        library_path = Path(workspace_path) / library_name
        if not library_path.exists():
            de.create_new_library(library_name, library_path)
            workspace.add_library(library_name, library_path, de.LibraryMode.SHARED)
            lib = workspace.open_library(library_name, library_path, de.LibraryMode.SHARED)
            lib.setup_schematic_tech()
        else:
            lib = workspace.open_library(library_name, library_path, de.LibraryMode.SHARED)
        
        # 创建原理图
        from keysight.ads.de import db_uu as db
        
        cell_name = f"{design_id}_schematic"
        design["cell_name"] = cell_name
        sch_design = db.create_schematic(f"{library_name}:{cell_name}:schematic")

        var_inst = sch_design.add_instance(
            ("ads_datacmps", "VAR", "symbol"),
            (3.5, -2.75),
            name="VAR1",
            angle=-90,
        )

        filter_band = design.get('filter_band', 'lowpass')

        if filter_band == 'highpass':
            # ---- HPF: 串联电容 + 并联电感 (与 LPF 镜像布局) ----
            C = design['C']
            L = design['L']
            fc = design['params']['fc']
            for i, c_val in enumerate(C):
                cap = sch_design.add_instance("ads_rflib:C:symbol", (i * 2, 0))
                cap.parameters["C"].value = f"Cs{i + 1} pF"
                cap.update_item_annotation()
                var_inst.vars[f"Cs{i + 1}"] = f"{c_val}"
                sch_design.add_wire([(i * 2 + 1, 0), (i * 2 + 2, 0)])
            for i, l_val in enumerate(L):
                ind = sch_design.add_instance("ads_rflib:L:symbol", (i * 2 + 1.5, -1), angle=-90)
                ind.parameters["L"].value = f"Lp{i + 1} nH"
                ind.update_item_annotation()
                var_inst.vars[f"Lp{i + 1}"] = f"{l_val}"
                sch_design.add_wire([(i * 2 + 1.5, 0), (i * 2 + 1.5, -1)])
                sch_design.add_instance("ads_rflib:GROUND:symbol", (i * 2 + 1.5, -2), angle=-90)
            x_pos = len(C) * 2
            stop_freq = max(fc * 3, 10e9)

        elif filter_band == 'bandpass':
            # ---- BPF: 串联LC谐振器 + 并联LC谐振器 ----
            L_series = design['L_series']
            C_series = design['C_series']
            L_shunt = design['L_shunt']
            C_shunt = design['C_shunt']
            fs_upper = design['params']['fs_upper']
            N = design['N']
            x_pos = 0
            ls_idx = 0
            lp_idx = 0
            for k in range(1, N + 1):
                if k % 2 != 0:
                    ind = sch_design.add_instance("ads_rflib:L:symbol", (x_pos, 0))
                    ind.parameters["L"].value = f"Ls{ls_idx + 1} nH"
                    ind.update_item_annotation()
                    var_inst.vars[f"Ls{ls_idx + 1}"] = f"{L_series[ls_idx]}"
                    sch_design.add_wire([(x_pos + 1, 0), (x_pos + 2, 0)])
                    cap = sch_design.add_instance("ads_rflib:C:symbol", (x_pos + 2, 0))
                    cap.parameters["C"].value = f"Cs{ls_idx + 1} pF"
                    cap.update_item_annotation()
                    var_inst.vars[f"Cs{ls_idx + 1}"] = f"{C_series[ls_idx]}"
                    sch_design.add_wire([(x_pos + 3, 0), (x_pos + 4, 0)])
                    ls_idx += 1
                    x_pos += 4
                else:
                    sch_design.add_wire([(x_pos, 0), (x_pos + 2, 0)])
                    cap = sch_design.add_instance("ads_rflib:C:symbol", (x_pos + 0.5, -1), angle=-90)
                    cap.parameters["C"].value = f"Cp{lp_idx + 1} pF"
                    cap.update_item_annotation()
                    var_inst.vars[f"Cp{lp_idx + 1}"] = f"{C_shunt[lp_idx]}"
                    sch_design.add_wire([(x_pos + 0.5, 0), (x_pos + 0.5, -1)])
                    ind = sch_design.add_instance("ads_rflib:L:symbol", (x_pos + 1.5, -1), angle=-90)
                    ind.parameters["L"].value = f"Lp{lp_idx + 1} nH"
                    ind.update_item_annotation()
                    var_inst.vars[f"Lp{lp_idx + 1}"] = f"{L_shunt[lp_idx]}"
                    sch_design.add_wire([(x_pos + 1.5, 0), (x_pos + 1.5, -1)])
                    sch_design.add_instance("ads_rflib:GROUND:symbol", (x_pos + 0.5, -2), angle=-90)
                    sch_design.add_instance("ads_rflib:GROUND:symbol", (x_pos + 1.5, -2), angle=-90)
                    lp_idx += 1
                    x_pos += 2
            stop_freq = max(fs_upper * 2.5, 10e9)

        else:
            # ---- LPF: 串联电感 + 并联电容 ----
            C = design['C']
            L = design['L']
            fs = design['params']['fs']
            for i, l_val in enumerate(L):
                ind = sch_design.add_instance("ads_rflib:L:symbol", (i * 2, 0))
                ind.parameters["L"].value = f"L{i + 1} nH"
                ind.update_item_annotation()
                var_inst.vars[f"L{i + 1}"] = f"{l_val}"
                sch_design.add_wire([(i * 2 + 1, 0), (i * 2 + 2, 0)])
            for i, c_val in enumerate(C):
                cap = sch_design.add_instance("ads_rflib:C:symbol", (i * 2 + 1.5, -1), angle=-90)
                cap.parameters["C"].value = f"C{i + 1} pF"
                cap.update_item_annotation()
                var_inst.vars[f"C{i + 1}"] = f"{c_val}"
                sch_design.add_wire([(i * 2 + 1.5, 0), (i * 2 + 1.5, -1)])
                sch_design.add_instance("ads_rflib:GROUND:symbol", (i * 2 + 1.5, -2), angle=-90)
            x_pos = len(L) * 2
            stop_freq = fs * 2

        del var_inst.vars["X"]

        # 端口
        sch_design.add_instance("ads_simulation:TermG:symbol", (-1, -1), angle=-90)
        sch_design.add_wire([(-1, -1), (-1, 0), (0, 0)])
        sch_design.add_instance("ads_simulation:TermG:symbol", (x_pos + 1, -1), angle=-90)
        sch_design.add_wire([(x_pos, 0), (x_pos + 1, 0.0)])
        sch_design.add_wire([(x_pos + 1, 0), (x_pos + 1, -1.0)])

        # S参数仿真
        sp = sch_design.add_instance("ads_simulation:S_Param:symbol", (2, 2))
        sp.parameters["Start"].value = "0.01 GHz"
        sp.parameters["Stop"].value = f"{stop_freq / 1e9} GHz"
        sp.parameters["Step"].value = "0.01 GHz"
        sp.update_item_annotation()
        
        sch_design.save_design()
        
        # 生成网表
        netlist = sch_design.generate_netlist()
        
        # 运行仿真
        from keysight.edatoolbox.ads import CircuitSimulator
        simulator = CircuitSimulator()
        
        sim_output_dir = output_base / design_id
        sim_output_dir.mkdir(parents=True, exist_ok=True)
        
        simulator.run_netlist(netlist, output_dir=str(sim_output_dir))
        
        # 提取结果
        ds_path = sim_output_dir / f"{cell_name}.ds"
        
        if not ds_path.exists():
            # 兼容不同名称的ds文件
            ds_candidates = list(sim_output_dir.glob("*.ds"))
            if ds_candidates:
                ds_path = ds_candidates[0]
            else:
                raise FileNotFoundError(f"未找到 .ds 数据集文件: {sim_output_dir}")
        
        output_data = dataset.open(Path(ds_path))
        
        # 提取S参数数据
        for block_name in output_data.varblock_names:
            if 'SP' in block_name.upper():
                df = output_data[block_name].to_dataframe().reset_index()
                
                # 保存数据
                csv_path = sim_output_dir / f"{block_name}.csv"
                df.to_csv(csv_path, index=False)
                
                # 计算性能指标
                import numpy as np

                filter_band = design.get('filter_band', 'lowpass')
                
                # 转换为dB
                S11_dB = 20 * np.log10(np.abs(df['S[1,1]'].values) + 1e-20)
                S21_dB = 20 * np.log10(np.abs(df['S[2,1]'].values) + 1e-20)
                freq = df['freq'].values

                if filter_band == 'highpass':
                    fc = design['params']['fc']
                    fs = design['params']['fs']
                    passband_mask = freq >= fc
                    stopband_mask = freq <= fs
                elif filter_band == 'bandpass':
                    f_lower = design['params']['f_lower']
                    f_upper = design['params']['f_upper']
                    fs_lower = design['params']['fs_lower']
                    fs_upper = design['params']['fs_upper']
                    passband_mask = (freq >= f_lower) & (freq <= f_upper)
                    stopband_mask = (freq <= fs_lower) | (freq >= fs_upper)
                else:
                    fc = design['params']['fc']
                    fs = design['params']['fs']
                    passband_mask = freq <= fc
                    stopband_mask = freq >= fs
                
                S11_passband = S11_dB[passband_mask]
                S21_passband = S21_dB[passband_mask]
                S21_stopband = S21_dB[stopband_mask]
                
                metrics = {
                    'S11_max_dB': float(np.max(S11_passband)) if len(S11_passband) > 0 else 0,
                    'S21_passband_min_dB': float(np.min(S21_passband)) if len(S21_passband) > 0 else 0,
                    'S21_passband_max_dB': float(np.max(S21_passband)) if len(S21_passband) > 0 else 0,
                    'S21_stopband_max_dB': float(np.max(S21_stopband)) if len(S21_stopband) > 0 else -100,
                    'passband_ripple_dB': float(np.max(S21_passband) - np.min(S21_passband)) if len(S21_passband) > 0 else 0,
                    'filter_band': filter_band,
                }
                if filter_band == 'bandpass':
                    metrics['f_center_Hz'] = design['params']['f_center']
                    metrics['bandwidth_Hz'] = design['params']['bandwidth']
                else:
                    metrics['fc_Hz'] = design['params'].get('fc', 0)
                    metrics['fs_Hz'] = design['params'].get('fs', 0)
                
                result = {
                    'design_id': design_id,
                    'design': design,
                    'metrics': metrics,
                    'data_path': str(csv_path),
                    'worker_id': worker_id
                }
                
                # 关闭workspace
                if de.workspace_is_open():
                    de.close_workspace()
                
                return result
        
        # 关闭workspace
        if de.workspace_is_open():
            de.close_workspace()
        
        raise RuntimeError(f"未找到S参数数据块: {design_id}")
        
    except Exception as e:
        # 关闭workspace
        if de.workspace_is_open():
            de.close_workspace()
        
        return {
            'design_id': design_id,
            'error': str(e),
            'worker_id': worker_id
        }
