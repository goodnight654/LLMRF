r"""
滤波器数据集生成与仿真批处理脚本
整合 filter_designer 和 ads_engine
自动生成滤波器设计并进行ADS仿真
python G:\wenlong\llmrf\adsapi\batch_filter_simulation.py --mode dataset    
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from .filter_designer import FilterDatasetGenerator, ChebyshevFilterDesigner, get_filter_designer
    from .ads_engine import ADSEngine
except ImportError:
    from filter_designer import FilterDatasetGenerator, ChebyshevFilterDesigner, get_filter_designer
    from ads_engine import ADSEngine

# 设置环境变量
os.environ['HPEESOF_DIR'] = r"C:/Program Files/Keysight/ADS2025_Update1"

from keysight.ads import de


# 随机样本数量（控制总样本数）
RANDOM_SPEC_COUNT = 15000

# 网格规格参数（控制网格样本数）
GRID_RIPPLE_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
GRID_FC_VALUES = [0.4e9, 0.6e9, 0.8e9, 1e9, 1.2e9, 1.5e9, 2e9, 2.5e9]
GRID_FS_RATIOS = [1.3, 1.5, 1.7, 2.0, 2.3, 2.5, 3.0]
GRID_LA_VALUES = [20, 25, 30, 35, 40, 45, 50, 55]
GRID_R0_VALUES = [25, 50, 75, 100]
GRID_ORDER_VALUES = [3, 4, 5, 6, 7, 8, 9]

# 批量仿真每批数量（只影响运行节奏）
SIM_BATCH_SIZE = 20

# 并行仿真配置（0 = 禁用并行，>0 = 并行进程数，-1 = 自动）
PARALLEL_WORKERS = -1  # 自动：使用 min(SIM_BATCH_SIZE, CPU核心数//2, 61)
MAX_WORKERS_LIMIT = 61  # Windows ProcessPoolExecutor 最大限制
# ==============================================================
# ==============================================================

class FilterSimulationPipeline:
    """滤波器设计+仿真流水线"""
    
    def __init__(self, 
                 workspace_path: str,
                 library_name: str = "filter_lib",
                 output_dir: str = "./filter_results"):
        """
        初始化流水线
        
        Args:
            workspace_path: ADS 工作空间路径
            library_name: 库名称
            output_dir: 输出目录
        """
        self.workspace_path = workspace_path
        self.library_name = library_name
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self.generator = FilterDatasetGenerator(
            output_dir=os.path.join(output_dir, "designs")
        )
        self.designer = ChebyshevFilterDesigner()
        
        print("="*70)
        print("滤波器仿真流水线")
        print("="*70)
        print(f"工作空间: {workspace_path}")
        print(f"库: {library_name}")
        print(f"输出目录: {output_dir}")
        print("="*70)
    
    def create_workspace_and_library(self):
        """创建ADS工作空间和库"""
        print("\n步骤 1: 创建ADS工作空间")
        
        # 检查工作空间是否存在
        if os.path.exists(self.workspace_path):
            print(f"  ℹ 工作空间已存在，将使用现有工作空间")
            if de.workspace_is_open():
                de.close_workspace()

            lib_defs = Path(self.workspace_path) / "lib.defs"
            try:
                if not lib_defs.exists():
                    raise RuntimeError("Library definition file not found")
                workspace = de.open_workspace(self.workspace_path)
            except RuntimeError:
                # 如果工作空间损坏/缺少 lib.defs，则重建
                import shutil
                try:
                    shutil.rmtree(self.workspace_path)
                except Exception:
                    pass
                workspace = de.create_workspace(self.workspace_path)
                workspace.open()
                print(f"  ✓ 重新创建工作空间: {self.workspace_path}")
        else:
            # 创建新工作空间
            workspace = de.create_workspace(self.workspace_path)
            workspace.open()
            print(f"  ✓ 创建新工作空间: {self.workspace_path}")
        
        # 检查库是否存在
        library_path = Path(self.workspace_path) / self.library_name
        
        if not library_path.exists():
            de.create_new_library(self.library_name, library_path)
            workspace.add_library(self.library_name, library_path, de.LibraryMode.SHARED)
            lib = workspace.open_library(self.library_name, library_path, de.LibraryMode.SHARED)
            lib.setup_schematic_tech()
            print(f"  ✓ 创建新库: {self.library_name}")
        else:
            lib = workspace.open_library(self.library_name, library_path, de.LibraryMode.SHARED)
            print(f"  ✓ 使用现有库: {self.library_name}")
        
        return workspace, lib

    def create_filter_schematic(self, lib: de.Library, design: Dict) -> Any:
        """在 ADS 中创建滤波器原理图。

        Args:
            lib: ADS 库对象
            design: 设计字典

        Returns:
            Any: ADS schematic 设计实例
        """
        from keysight.ads.de import db_uu as db

        design_id = design['id']
        C = design['C']
        fs = design['params']['fs']
        L = design['L']

        # 创建原理图
        cell_name = f"{design_id}_schematic"
        design["cell_name"] = cell_name
        sch_design = db.create_schematic(f"{self.library_name}:{cell_name}:schematic")

        # 添加变量
        var_inst = sch_design.add_instance(
            ("ads_datacmps", "VAR", "symbol"),
            (3.5, -2.75),
            name="VAR1",
            angle=-90,
        )
        
        # 添加电感
        for i, l_val in enumerate(L):
            ind = sch_design.add_instance("ads_rflib:L:symbol", (i * 2, 0))
            ind.parameters["L"].value = f"L{i + 1} nH"
            ind.update_item_annotation()
            var_inst.vars[f"L{i + 1}"] = f"{l_val}"
            sch_design.add_wire([(i * 2 + 1, 0), (i * 2 + 2, 0)])
        
        # 添加电容
        for i, c_val in enumerate(C):
            cap = sch_design.add_instance(
                "ads_rflib:C:symbol", 
                (i * 2 + 1.5, -1), 
                angle=-90
            )
            cap.parameters["C"].value = f"C{i + 1} pF"
            cap.update_item_annotation()
            var_inst.vars[f"C{i + 1}"] = f"{c_val}"
            sch_design.add_wire([(i * 2 + 1.5, 0), (i * 2 + 1.5, -1)])
            sch_design.add_instance(
                "ads_rflib:GROUND:symbol", 
                (i * 2 + 1.5, -2), 
                angle=-90
            )
        
        del var_inst.vars["X"]
        
        # 添加端口
        sch_design.add_instance("ads_simulation:TermG:symbol", (-1, -1), angle=-90)
        sch_design.add_wire([(-1, -1), (-1, 0), (0, 0)])
        
        sch_design.add_instance(
            "ads_simulation:TermG:symbol", 
            (len(L) * 2 + 1, -1), 
            angle=-90
        )
        sch_design.add_wire([(len(L) * 2, 0), (len(L) * 2 + 1, 0.0)])
        sch_design.add_wire([(len(L) * 2 + 1, 0), (len(L) * 2 + 1, -1.0)])
        
        # 添加S参数仿真
        sp = sch_design.add_instance("ads_simulation:S_Param:symbol", (2, 2))
        sp.parameters["Start"].value = "0.01 GHz"
        sp.parameters["Stop"].value = f"{(fs * 2) / 1e9} GHz"
        sp.parameters["Step"].value = "0.01 GHz"
        sp.update_item_annotation()
        
        sch_design.save_design()
        
        return sch_design
    
    def run_batch_simulation(self, 
                            designs: List[Dict],
                            max_designs: int = None) -> List[Dict]:
        """
        批量运行仿真
        
        Args:
            designs: 设计列表
            max_designs: 最大仿真数量（用于测试）
            
        Returns:
            list: 仿真结果列表
        """
        if max_designs:
            designs = designs[:max_designs]

        total = len(designs)
        print(f"\n步骤 2: 批量仿真 ({total} 个设计)")
        start_time = time.time()

        # 创建工作空间
        workspace, lib = self.create_workspace_and_library()

        results = []

        for i, design in enumerate(designs):
            item_start = time.time()
            design_id = design['id']
            idx = i + 1
            progress = (idx / total) * 100 if total else 100
            elapsed = time.time() - start_time
            avg = elapsed / idx if idx else 0
            eta = avg * (total - idx) if total else 0
            print(
                f"\n  [{idx}/{total}] 仿真: {design_id} | "
                f"进度 {progress:.1f}% | 已用 {elapsed:.1f}s | 预计剩余 {eta:.1f}s"
            )

            try:
                # 创建原理图
                sch_design = self.create_filter_schematic(lib, design)
                print("    ✓ 原理图已创建")

                # 生成网表
                netlist = sch_design.generate_netlist()
                print(f"    ✓ 网表已生成 ({len(netlist)} 字符)")

                # 运行仿真
                from keysight.edatoolbox.ads import CircuitSimulator
                simulator = CircuitSimulator()

                sim_output_dir = os.path.join(
                    self.output_dir,
                    "simulations",
                    design_id
                )
                os.makedirs(sim_output_dir, exist_ok=True)

                simulator.run_netlist(netlist, output_dir=sim_output_dir)
                item_elapsed = time.time() - item_start
                print(f"    ✓ 仿真完成 (用时 {item_elapsed:.1f}s)")

                # 提取结果
                from keysight.ads import dataset
                cell_name = design.get("cell_name", f"{design_id}_schematic")
                ds_path = os.path.join(sim_output_dir, f"{cell_name}.ds")

                if not os.path.exists(ds_path):
                    # 兼容 ADS 可能生成不同名称的 ds 文件
                    ds_candidates = list(Path(sim_output_dir).glob("*.ds"))
                    if ds_candidates:
                        ds_path = str(ds_candidates[0])
                        print(f"    ℹ 未找到预期数据集，改用: {ds_path}")
                    else:
                        raise FileNotFoundError(f"未找到 .ds 数据集文件: {sim_output_dir}")

                output_data = dataset.open(Path(ds_path))

                # 提取S参数数据
                for block_name in output_data.varblock_names:
                    if 'SP' in block_name.upper():
                        df = output_data[block_name].to_dataframe().reset_index()

                        # 保存数据
                        csv_path = os.path.join(sim_output_dir, f"{block_name}.csv")
                        df.to_csv(csv_path, index=False)

                        # 计算性能指标
                        metrics = self.calculate_metrics(df, design)

                        result = {
                            'design_id': design_id,
                            'design': design,
                            'metrics': metrics,
                            'data_path': csv_path
                        }
                        results.append(result)

                        print("    ✓ 指标提取完成")
                        print(f"      - S11_max: {metrics['S11_max_dB']:.2f} dB")
                        print(f"      - S21_min: {metrics['S21_passband_min_dB']:.2f} dB")
                        break

            except Exception as e:
                print(f"    ✗ 仿真失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ 批量仿真完成: {len(results)}/{len(designs)}")

        # 关闭工作空间
        if de.workspace_is_open():
            de.close_workspace()

        return results
    
    def run_parallel_batch_simulation(self, 
                                       designs: List[Dict],
                                       max_workers: int = None) -> List[Dict]:
        """
        并行批量仿真（利用多核CPU加速）
        
        Args:
            designs: 设计列表
            max_workers: 并行worker数量（None=自动检测CPU核心数-1）
            
        Returns:
            list: 仿真结果列表
        """
        try:
            from adsapi.batch_filter_simulation_parallel import simulate_single_design
        except ImportError:
            from batch_filter_simulation_parallel import simulate_single_design
        
        total = len(designs)
        
        # 确定worker数量
        if max_workers is None or max_workers < 0:
            # 自动检测：优先匹配批次大小，但不超过CPU核心数一半和Windows限制
            cpu_count = multiprocessing.cpu_count()
            # 使用批次大小作为首选，这样每批可以完全并行化
            max_workers = min(len(designs), cpu_count // 2, MAX_WORKERS_LIMIT)
            max_workers = max(1, max_workers)  # 至少使用1个
        elif max_workers == 0:
            # 0表示禁用并行，使用串行仿真
            return self.run_batch_simulation(designs)
        elif max_workers > MAX_WORKERS_LIMIT:
            # 超过Windows限制，自动限制
            print(f"  ⚠ 警告: workers={max_workers} 超过Windows限制，自动调整为 {MAX_WORKERS_LIMIT}")
            max_workers = MAX_WORKERS_LIMIT
        
        print(f"\n步骤 2: 并行批量仿真 ({total} 个设计, {max_workers} 个并行进程)")
        start_time = time.time()
        
        # 准备参数
        args_list = [
            (design, self.workspace_path, self.library_name)
            for design in designs
        ]
        
        results = []
        completed = 0
        failed = 0
        
        # 使用进程池并行仿真
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_design = {
                executor.submit(simulate_single_design, args): args[0]
                for args in args_list
            }
            
            # 收集结果
            for future in as_completed(future_to_design):
                design = future_to_design[future]
                try:
                    result = future.result()
                    
                    if 'error' in result:
                        failed += 1
                        print(f"  ✗ [{completed+failed}/{total}] {result['design_id']}: {result['error']}")
                    else:
                        completed += 1
                        results.append(result)
                        
                        # 打印进度
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (completed + failed) if (completed + failed) > 0 else 0
                        eta = avg_time * (total - completed - failed)
                        progress = ((completed + failed) / total) * 100
                        
                        print(
                            f"  ✓ [{completed+failed}/{total}] {result['design_id']} | "
                            f"进度 {progress:.1f}% | 已用 {elapsed:.1f}s | "
                            f"预计剩余 {eta:.1f}s | Worker {result.get('worker_id', '?')}"
                        )
                        
                except Exception as e:
                    failed += 1
                    print(f"  ✗ [{completed+failed}/{total}] {design['id']}: 进程异常 - {e}")
        
        total_time = time.time() - start_time
        print(f"\n✓ 并行仿真完成: 成功 {completed}/{total}, 失败 {failed}/{total}")
        print(f"✓ 总耗时: {total_time:.1f}s, 平均速度: {total_time/total:.2f}s/个")
        if max_workers > 1:
            serial_estimate = total_time * max_workers
            speedup = serial_estimate / total_time if total_time > 0 else 1
            print(f"✓ 加速比: {speedup:.1f}x (相比串行预计节省 {(serial_estimate - total_time)/60:.1f} 分钟)")

        return results
    
    def calculate_metrics(self, df: pd.DataFrame, design: Dict) -> Dict:
        """
        计算滤波器性能指标
        
        Args:
            df: S参数数据
            design: 设计参数
            
        Returns:
            dict: 性能指标
        """
        import numpy as np
        
        fc = design['params']['fc']
        fs = design['params']['fs']
        
        # 转换为dB
        S11_dB = 20 * np.log10(np.abs(df['S[1,1]'].values) + 1e-20)
        S21_dB = 20 * np.log10(np.abs(df['S[2,1]'].values) + 1e-20)
        freq = df['freq'].values
        
        # 通带指标 (0 - fc)
        passband_mask = freq <= fc
        S11_passband = S11_dB[passband_mask]
        S21_passband = S21_dB[passband_mask]
        
        # 阻带指标 (fs - end)
        stopband_mask = freq >= fs
        S21_stopband = S21_dB[stopband_mask]
        
        metrics = {
            'S11_max_dB': float(np.max(S11_passband)) if len(S11_passband) > 0 else 0,
            'S21_passband_min_dB': float(np.min(S21_passband)) if len(S21_passband) > 0 else 0,
            'S21_passband_max_dB': float(np.max(S21_passband)) if len(S21_passband) > 0 else 0,
            'S21_stopband_max_dB': float(np.max(S21_stopband)) if len(S21_stopband) > 0 else -100,
            'passband_ripple_dB': float(np.max(S21_passband) - np.min(S21_passband)) if len(S21_passband) > 0 else 0,
            'fc_Hz': fc,
            'fs_Hz': fs
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], filename: str = "simulation_results.json", save_summary: bool = True):
        """保存仿真结果
        
        Args:
            results: 结果列表
            filename: JSON文件名
            save_summary: 是否保存summary.csv（避免中间批次覆盖）
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存: {filepath}")
        print(f"✓ 当前样本数: {len(results)}")
        
        # 创建摘要表格（仅在需要时保存，避免中间批次覆盖）
        if save_summary:
            summary = []
            for r in results:
                summary.append({
                    'design_id': r['design_id'],
                    'order': r['design']['N'],
                    'fc_MHz': r['design']['params']['fc'] / 1e6,
                    'fs_MHz': r['design']['params']['fs'] / 1e6,
                    'S11_max_dB': r['metrics']['S11_max_dB'],
                    'S21_min_dB': r['metrics']['S21_passband_min_dB'],
                    'ripple_dB': r['metrics']['passband_ripple_dB']
                })
            
            summary_df = pd.DataFrame(summary)
            summary_path = os.path.join(self.output_dir, "summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            print(f"✓ 摘要表格已保存: {summary_path}")
            print(f"\n{summary_df.to_string(index=False)}")


def main_quick_test():
    """快速测试 - 生成并仿真3个滤波器"""
    print("="*70)
    print("快速测试 - 3个滤波器设计+仿真")
    print("="*70)
    
    pipeline = FilterSimulationPipeline(
        workspace_path=r"G:\wenlong\ADS\test_Filter",
        library_name="filter_test_lib",
        output_dir=r"G:\wenlong\ADS\test_Filter\results"
    )
    
    # 生成3个测试设计
    specs = [
        {'id': 'test_800M', 'ripple_db': 0.1, 'fc': 800e6, 'fs': 1500e6, 'R0': 50, 'La': 40},
        {'id': 'test_1G', 'ripple_db': 0.5, 'fc': 1e9, 'fs': 1.8e9, 'R0': 50, 'La': 35},
        {'id': 'test_1_5G', 'ripple_db': 0.1, 'fc': 1.5e9, 'fs': 2.5e9, 'R0': 50, 'La': 45},
    ]
    
    designs = pipeline.generator.design_batch(specs)
    results = pipeline.run_batch_simulation(designs)
    pipeline.save_results(results)


def main_dataset_generation(filter_type: str, workers: int = None):
    """大规模数据集生成
    
    Args:
        filter_type: 滤波器类型
        workers: 并行进程数（None=自动检测，0=禁用并行）
    """
    print("="*70)
    print("大规模数据集生成")
    print("="*70)
    date_tag = time.strftime("%Y%m%d")
    pipeline = FilterSimulationPipeline(
        workspace_path=rf"G:\wenlong\ADS\filter_dataset_wrk_{date_tag}",
        library_name=f"filter_dataset_lib_{date_tag}",
        output_dir=rf"G:\wenlong\ADS\filter_dataset_results_{date_tag}",
    )
    pipeline.generator.filter_type = filter_type
    pipeline.generator.designer = get_filter_designer(filter_type)
    def _spec_key(spec: Dict) -> Tuple:
        return (
            spec.get('filter_type', 'chebyshev'),
            int(spec.get('order') or 0),
            round(spec['ripple_db'], 4),
            int(spec['fc']),
            int(spec['fs']),
            int(spec['R0']),
            int(spec['La'])
        )

    def _unique_specs(specs: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for s in specs:
            key = _spec_key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _random_specs(count: int) -> List[Dict]:
        return pipeline.generator.generate_random_specs(
            num_samples=count,
            R0_values=GRID_R0_VALUES,
            order_values=GRID_ORDER_VALUES,
            filter_type=filter_type,
        )
    grid_specs = pipeline.generator.generate_grid_specs(
        ripple_values=GRID_RIPPLE_VALUES,
        fc_values=GRID_FC_VALUES,
        fs_ratios=GRID_FS_RATIOS,
        La_values=GRID_LA_VALUES,
        R0_values=GRID_R0_VALUES,
        order_values=GRID_ORDER_VALUES,
        filter_type=filter_type,
    )

    random_specs = _random_specs(RANDOM_SPEC_COUNT)
    # 生成全局唯一 design_id，避免 batch 内从 0000 重新计数
    specs = _unique_specs(grid_specs + random_specs)
    for idx, spec in enumerate(specs):
        ftype = spec.get("filter_type", "chebyshev")
        spec["id"] = f"ds_{date_tag}_{ftype}_{idx:06d}"

    designs = pipeline.generator.design_batch(specs)

    # 自动选择串行/并行仿真
    if workers is None:
        workers = PARALLEL_WORKERS
    
    # 确定worker数量
    if workers < 0:
        cpu_count = multiprocessing.cpu_count()
        # 优先使用批次大小，这样每批可以完全并行化
        workers = min(SIM_BATCH_SIZE, cpu_count // 2, MAX_WORKERS_LIMIT)
        workers = max(1, workers)
    elif workers > MAX_WORKERS_LIMIT:
        print(f"\n⚠ 警告: workers={workers} 超过Windows限制，自动调整为 {MAX_WORKERS_LIMIT}")
        workers = MAX_WORKERS_LIMIT
    
    use_parallel = workers > 0
    
    if use_parallel:
        print(f"ℹ 使用并行仿真: {workers} 个进程 (CPU核心数: {multiprocessing.cpu_count()})")
    else:
        print("ℹ 使用串行仿真")
    print("="*70)

    # 批量仿真（可以分批进行）
    batch_size = SIM_BATCH_SIZE
    all_results = []
    
    for i in range(0, len(designs), batch_size):
        batch = designs[i:i+batch_size]
        print(f"\n{'='*70}")
        print(f"仿真批次 {i//batch_size + 1}/{(len(designs)-1)//batch_size + 1}")
        print(f"{'='*70}")
        
        # 根据配置选择串行或并行
        if use_parallel:
            results = pipeline.run_parallel_batch_simulation(batch, max_workers=workers)
        else:
            results = pipeline.run_batch_simulation(batch)
        
        all_results.extend(results)
        
        # 保存中间结果（仅当前批次JSON，不保存summary避免覆盖）
        pipeline.save_results(results, f"results_batch_{i//batch_size + 1}.json", save_summary=False)
    
    # 保存最终结果（包含完整summary.csv）
    pipeline.save_results(all_results, "final_results.json", save_summary=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='滤波器数据集生成工具')
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['test', 'dataset'],
        help='运行模式: test/dataset'
    )
    parser.add_argument(
        '--filter_type',
        type=str,
        default='chebyshev',
        choices=['chebyshev', 'butterworth', 'elliptic'],
        help='滤波器类型: chebyshev/butterworth/elliptic'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并行进程数（0=禁用并行，-1=自动检测CPU核心数-1，默认使用配置文件设置）'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        main_quick_test()
    else:
        main_dataset_generation(args.filter_type, workers=args.workers)
