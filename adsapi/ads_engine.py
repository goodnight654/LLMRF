"""
ADS 仿真引擎核心模块 (无 GUI 版本)

专门设计用于在 ADS Python 环境中运行
支持命令行调用和 LLM 接口集成

使用方法：
    from ads_engine import ADSEngine
    engine = ADSEngine()
    engine.connect(workspace_path, library_name, design_name)
    results = engine.run_simulation()
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

# 导入 keysight ADS 模块（注意顺序：dds 必须在 dataset 之前）
try:
    from keysight.ads import de
    from keysight.ads.de import db_uu as db
    from keysight.ads import dds  # 必须先导入 dds
    from keysight.ads import dataset  # 再导入 dataset
    from keysight.edatoolbox import ads
    from keysight.edatoolbox import util
    print("✓ Keysight ADS 模块加载成功")
except ImportError as e:
    print(f"✗ 无法导入 Keysight 模块: {e}")
    print("  请确保在 ADS Python 环境中运行此脚本")
    raise


class ADSEngine:
    """ADS 仿真引擎 - 核心驱动类"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化 ADS 引擎
        
        Args:
            verbose: 是否输出详细日志
        """
        self.workspace = None
        self.library = None
        self.library_name = None
        self.design = None
        self.design_name = None
        self.connected = False
        self.verbose = verbose
        
        # 仿真结果缓存
        self.netlist_cache = None
        self.last_output_dir = None
        self.last_ds_path = None
        self.last_results = None
        self.last_block_names = None
        
        self._log("ADS 引擎已初始化")
    
    def _log(self, message: str, level: str = "INFO"):
        """内部日志函数"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ",
                "SUCCESS": "✓",
                "ERROR": "✗",
                "WARNING": "⚠"
            }.get(level, "•")
            print(f"{prefix} {message}")
    
    def connect(self, workspace_path: str, library_name: str, design_name: str) -> bool:
        """
        连接到 ADS 工作空间和设计
        
        Args:
            workspace_path: 工作空间路径
            library_name: 库名称
            design_name: 设计名称
            
        Returns:
            bool: 连接是否成功
        """
        try:
            self._log("=" * 60)
            self._log(f"连接到 ADS 设计")
            self._log(f"  工作空间: {workspace_path}")
            self._log(f"  库: {library_name}")
            self._log(f"  设计: {design_name}")
            self._log("=" * 60)
            
            # Step 1: 获取或打开工作空间
            if de.workspace_is_open():
                self.workspace = de.active_workspace()
                self._log(f"使用当前活动工作空间: {self.workspace}", "SUCCESS")
            else:
                if workspace_path and os.path.exists(workspace_path):
                    self._log(f"打开工作空间: {workspace_path}")
                    self.workspace = de.open_workspace(workspace_path)
                    self._log("工作空间已打开", "SUCCESS")
                else:
                    self._log("工作空间路径无效或未指定", "ERROR")
                    return False
            
            # Step 2: 检查库
            if de.library_is_open(library_name):
                self._log(f"库 '{library_name}' 已打开", "SUCCESS")
                self.library_name = library_name
            else:
                # 尝试查找可用库
                try:
                    available_libs = list(self.workspace.libraries)
                    lib_names = [lib.name for lib in available_libs]
                    
                    if library_name in lib_names:
                        self.library_name = library_name
                        self._log(f"找到库: {library_name}", "SUCCESS")
                    elif lib_names:
                        self.library_name = lib_names[0]
                        self._log(f"未找到库 '{library_name}'，使用: {self.library_name}", "WARNING")
                    else:
                        self._log("工作空间中没有可用的库", "ERROR")
                        return False
                except Exception as e:
                    self._log(f"获取库列表失败: {e}", "ERROR")
                    return False
            
            # Step 3: 打开设计
            try:
                cell_view_name = f"{self.library_name}:{design_name}:schematic"
                self._log(f"打开设计: {cell_view_name}")
                
                self.design = db.open_design(cell_view_name, db.DesignMode.READ_ONLY)
                self.design_name = design_name
                self._log(f"设计已打开: {design_name}", "SUCCESS")
                
            except Exception as e:
                self._log(f"打开设计失败: {e}", "ERROR")
                # 尝试列出可用设计
                try:
                    for lib in self.workspace.libraries:
                        if lib.name == self.library_name:
                            designs = [d.name for d in lib.designs]
                            self._log(f"可用设计: {designs}", "WARNING")
                            if designs:
                                self.design_name = designs[0]
                                cell_view_name = f"{self.library_name}:{self.design_name}:schematic"
                                self.design = db.open_design(cell_view_name, db.DesignMode.READ_ONLY)
                                self._log(f"使用设计: {self.design_name}", "SUCCESS")
                            break
                except Exception as e2:
                    self._log(f"获取设计列表失败: {e2}", "ERROR")
                    return False
            
            self.connected = True
            self._log("=" * 60)
            self._log("ADS 连接成功！", "SUCCESS")
            self._log("=" * 60)
            return True
            
        except Exception as e:
            self._log(f"连接失败: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def generate_netlist(self) -> Optional[str]:
        """
        从当前设计生成网表
        
        Returns:
            str: 网表文本，失败返回 None
        """
        if not self.design:
            self._log("没有打开的设计，无法生成网表", "ERROR")
            return None
        
        try:
            self._log(f"正在从设计 '{self.design_name}' 生成网表...")
            netlist = self.design.generate_netlist()
            self.netlist_cache = netlist
            self._log(f"网表生成成功，长度: {len(netlist)} 字符", "SUCCESS")
            return netlist
        except Exception as e:
            self._log(f"生成网表失败: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return None
    
    def update_netlist_parameters(self, netlist: str, parameters: Dict[str, float]) -> str:
        """
        更新网表中的参数
        
        Args:
            netlist: 原始网表文本
            parameters: 参数字典 {参数名: 新值}
            
        Returns:
            str: 更新后的网表
        """
        updated_netlist = netlist
        
        for param_name, new_value in parameters.items():
            # 匹配参数赋值模式：参数名=数值
            pattern = rf'({param_name}\s*=\s*)([\d.eE+-]+)'
            
            def replace_value(match):
                return f"{match.group(1)}{new_value}"
            
            updated_netlist = re.sub(pattern, replace_value, updated_netlist)
            self._log(f"更新参数: {param_name} = {new_value}")
        
        return updated_netlist
    
    def run_simulation(self, output_no: int = 99, parameters: Optional[Dict[str, float]] = None) -> Tuple[Optional[List[str]], Optional[Dict[int, Any]]]:
        """
        运行仿真
        
        Args:
            output_no: 输出编号（用于生成唯一的输出文件夹）
            parameters: 可选的参数字典，用于更新网表参数
            
        Returns:
            (ds_path, dataframe_dict): 数据集路径和结果字典
        """
        if not self.design:
            self._log("没有打开的设计，无法运行仿真", "ERROR")
            return None, None
        
        try:
            self._log("=" * 60)
            self._log(f"开始仿真 (输出编号: {output_no})")
            self._log("=" * 60)
            
            # 生成网表
            netlist = self.generate_netlist()
            if not netlist:
                return None, None
            
            # 如果提供了参数，更新网表
            if parameters:
                netlist = self.update_netlist_parameters(netlist, parameters)
            
            # 准备仿真工作目录
            if de.workspace_is_open():
                ws = de.active_workspace()
                ws_path = str(ws.path)
            else:
                ws_path = os.getcwd()
            
            # 创建输出目录
            output_dir = os.path.join(ws_path, f"sim_output_{output_no}")
            os.makedirs(output_dir, exist_ok=True)
            self._log(f"输出目录: {output_dir}")
            
            # 保存网表到临时文件
            netlist_file = os.path.join(output_dir, "netlist.net")
            with open(netlist_file, 'w', encoding='utf-8') as f:
                f.write(netlist)
            self._log(f"网表已保存: {netlist_file}")
            
            # 运行仿真（使用 CircuitSimulator）
            self._log("调用仿真引擎...")
            simulator = ads.CircuitSimulator()
            simulator.run_netlist(netlist, output_dir=output_dir)
            self._log("仿真命令已发送", "SUCCESS")
            
            # 等待仿真完成
            time.sleep(2)
            
            # 提取仿真结果
            ds_path = os.path.join(output_dir, f"{self.design_name}.ds")
            results = self._extract_simulation_results(ds_path)
            
            # 缓存结果
            self.last_output_dir = output_dir
            self.last_ds_path = ds_path
            self.last_results = results
            
            if results:
                self._log(f"仿真成功！找到 {len(results)} 个数据块", "SUCCESS")
                for name in results.keys():
                    self._log(f"  - {name}")
            else:
                self._log("仿真可能失败或无结果", "WARNING")
            
            self._log("=" * 60)
            return ds_path, results
            
        except Exception as e:
            self._log(f"仿真失败: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return None, None
    
    def _extract_simulation_results(self, ds_path: str) -> Optional[Dict]:
        """
        从 .ds 文件中提取仿真结果
        
        Args:
            ds_path: .ds 文件路径
            
        Returns:
            dict: {数据块名: DataFrame}
        """
        try:
            if not os.path.exists(ds_path):
                self._log(f".ds 文件不存在: {ds_path}", "WARNING")
                return None
            
            self._log("正在提取仿真结果...")
            
            # 使用 dataset 打开数据文件
            output_data = dataset.open(Path(ds_path))
            
            # 获取所有数据块名称
            block_names = output_data.varblock_names
            self._log(f"找到数据块: {block_names}")
            
            # 提取每个数据块为 DataFrame
            results = {}
            for block_name in block_names:
                try:
                    df = output_data[block_name].to_dataframe().reset_index()
                    results[block_name] = df
                    self._log(f"提取数据块 {block_name}: {df.shape}")
                except Exception as e:
                    self._log(f"提取数据块 {block_name} 失败: {e}", "WARNING")
            
            return results
            
        except Exception as e:
            self._log(f"提取结果失败: {e}", "ERROR")
            return None
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取引擎当前状态
        
        Returns:
            dict: 状态信息
        """
        return {
            'connected': self.connected,
            'workspace': str(self.workspace) if self.workspace else None,
            'library': self.library_name,
            'design': self.design_name,
            'has_netlist': self.netlist_cache is not None,
            'has_results': self.last_results is not None,
            'output_dir': self.last_output_dir
        }
    
    def close(self):
        """关闭连接，清理资源"""
        self._log("关闭 ADS 引擎")
        self.design = None
        self.library = None
        self.workspace = None
        self.connected = False


if __name__ == "__main__":
    # 测试代码
    print("ADS Engine 模块测试")
    print("=" * 60)
    
    # 检查是否在 ADS 环境中
    try:
        if de.workspace_is_open():
            ws = de.active_workspace()
            print(f"✓ 检测到活动工作空间: {ws}")
            
            # 创建引擎实例
            engine = ADSEngine(verbose=True)
            
            # 获取状态
            state = engine.get_current_state()
            print("\n当前状态:")
            print(json.dumps(state, indent=2))
        else:
            print("⚠ 未检测到活动工作空间")
            print("  请在 ADS 中打开一个工作空间后再运行")
    except Exception as e:
        print(f"✗ 错误: {e}")
