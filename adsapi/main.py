"""
ADS 自动化仿真主程序（命令行版本）

无 GUI，支持命令行参数调用
集成 LLM 辅助设计功能
保留图表生成能力

使用方法：
    python main.py --workspace <workspace_path> --library <lib_name> --design <design_name>
    
或在 ADS Python 环境中：
    python main.py --auto
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# 首先加载配置并设置环境变量
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = json.load(f)
        ads_install_path = _config['ads']['install_path']
        os.environ['HPEESOF_DIR'] = ads_install_path
        print(f"✓ 已从配置文件设置 HPEESOF_DIR: {ads_install_path}")

# 设置 ADS 环境变量（如果未设置）
if 'HPEESOF_DIR' not in os.environ:
    # 根据用户的 ADS 安装路径设置
    ads_install_path = r"F:/Program Files (x86)/ADS2026"
    if os.path.exists(ads_install_path):
        os.environ['HPEESOF_DIR'] = ads_install_path
        print(f"✓ 已设置 HPEESOF_DIR: {ads_install_path}")
    else:
        print(f"⚠ 警告：未找到 ADS 安装路径: {ads_install_path}")

# 导入自定义模块
try:
    from ads_engine import ADSEngine
    from netlist_parser import NetlistParser
    from visualizer import ResultVisualizer
    from llm_interface import LLMInterface, MockLLMInterface
    from post_processor import PostProcessor, FormulaCalculator
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    print("  请确保所有模块都在 adsapi 目录下")
    sys.exit(1)


class ADSAutomation:
    """ADS 自动化仿真主控制器"""
    
    def __init__(self, 
                 workspace_path: str,
                 library_name: str,
                 design_name: str,
                 output_dir: str = "./results",
                 use_llm: bool = False,
                 llm_config: Optional[Dict] = None):
        """
        初始化自动化控制器
        
        Args:
            workspace_path: ADS 工作空间路径
            library_name: 库名称
            design_name: 设计名称
            output_dir: 结果输出目录
            use_llm: 是否使用 LLM
            llm_config: LLM 配置字典
        """
        self.workspace_path = workspace_path
        self.library_name = library_name
        self.design_name = design_name
        # 转换为绝对路径
        self.output_dir = os.path.abspath(output_dir)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.engine = ADSEngine(verbose=True)
        self.visualizer = ResultVisualizer(output_dir=os.path.join(output_dir, "plots"))
        self.post_processor = PostProcessor()
        
        # LLM 接口
        if use_llm and llm_config:
            self.llm = LLMInterface(**llm_config)
        elif use_llm:
            # 使用默认配置
            self.llm = LLMInterface(
                model_type="local",
                api_url="http://localhost:11434/api/generate",
                model_name="qwen2.5:14b"
            )
        else:
            # 使用 Mock LLM（测试模式）
            self.llm = MockLLMInterface()
        
        self.variables = []
        self.netlist = None
        
        print("=" * 70)
        print("ADS 自动化仿真系统")
        print("=" * 70)
        print(f"工作空间: {workspace_path}")
        print(f"库: {library_name}")
        print(f"设计: {design_name}")
        print(f"输出目录: {output_dir}")
        print(f"LLM 支持: {'启用' if use_llm else '测试模式'}")
        print("=" * 70)
    
    def connect(self) -> bool:
        """连接到 ADS"""
        print("\n步骤 1: 连接到 ADS")
        return self.engine.connect(self.workspace_path, self.library_name, self.design_name)
    
    def load_design(self) -> bool:
        """加载设计并解析变量"""
        print("\n步骤 2: 加载设计并解析变量")
        
        # 生成网表
        self.netlist = self.engine.generate_netlist()
        if not self.netlist:
            print("✗ 无法生成网表")
            return False
        
        # 保存网表到文件
        netlist_path = os.path.join(self.output_dir, "design.net")
        with open(netlist_path, 'w', encoding='utf-8') as f:
            f.write(self.netlist)
        print(f"✓ 网表已保存: {netlist_path}")
        
        # 解析变量
        self.variables = NetlistParser.parse_variables(self.netlist)
        if self.variables:
            print(f"✓ 找到 {len(self.variables)} 个优化变量:")
            print(NetlistParser.format_variable_table(self.variables))
            
            # 保存变量信息
            vars_path = os.path.join(self.output_dir, "variables.json")
            with open(vars_path, 'w', encoding='utf-8') as f:
                json.dump(self.variables, f, indent=2)
            print(f"✓ 变量信息已保存: {vars_path}")
            return True
        else:
            print("⚠ 未找到优化变量，将使用当前参数运行仿真")
            return True
    
    def run_baseline_simulation(self) -> bool:
        """运行基线仿真"""
        print("\n步骤 3: 运行基线仿真")
        
        ds_path, results = self.engine.run_simulation(output_no=1)
        
        if results:
            print(f"✓ 基线仿真完成")
            print(f"  数据集: {ds_path}")
            print(f"  数据块数量: {len(results)}")
            
            # 保存结果
            results_path = os.path.join(self.output_dir, "baseline_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'ds_path': str(ds_path),
                    'blocks': list(results.keys()),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            print(f"✓ 基线结果已保存: {results_path}")
            
            # 绘制结果
            self._plot_results(results)
            
            return True
        else:
            print("✗ 基线仿真失败")
            return False
    
    def _plot_results(self, results: Dict):
        """绘制仿真结果和生成表格"""
        print("\n处理仿真结果...")
        
        for block_name, df in results.items():
            print(f"\n  处理数据块: {block_name}")
            
            # 检查是否有 S 参数
            s_param_cols = [col for col in df.columns if col.startswith('S[') or col.startswith('S(')]
            
            if s_param_cols:
                # 只有S参数才画图
                print(f"    绘制 S 参数图...")
                self.visualizer.plot_from_dataframe(
                    df,
                    freq_col='freq',
                    data_cols=s_param_cols,
                    title=f"S-Parameters - {block_name}",
                    filename=f"s_params_{block_name}.png",
                    to_db=True
                )
            
            # 检查是否是HB数据（用于后处理）
            if 'HB' in block_name.upper():
                print(f"    检测到 HB 数据，执行后处理...")
                
                # 执行后处理计算
                processed_df = self.post_processor.process_hb_data(df)
                
                if not processed_df.empty:
                    # 保存后处理结果
                    processed_path = os.path.join(self.output_dir, f"processed_{block_name}.csv")
                    processed_df.to_csv(processed_path, index=False)
                    print(f"    ✓ 后处理结果已保存: {processed_path}")
                    
                    # 绘制后处理结果（功率、PAE等）
                    plot_cols = [col for col in processed_df.columns if col != 'freq']
                    if plot_cols:
                        # 功率相关
                        power_cols = [col for col in plot_cols if 'P_' in col or 'Gain' in col]
                        if power_cols:
                            self.visualizer.plot_from_dataframe(
                                processed_df,
                                freq_col='freq',
                                data_cols=power_cols,
                                title=f"Power Analysis - {block_name}",
                                filename=f"power_{block_name}.png",
                                ylabel="Power (dBm) / Gain (dB)",
                                to_db=False
                            )
                        
                        # 效率相关
                        eff_cols = [col for col in plot_cols if 'PAE' in col or 'Efficiency' in col]
                        if eff_cols:
                            self.visualizer.plot_from_dataframe(
                                processed_df,
                                freq_col='freq',
                                data_cols=eff_cols,
                                title=f"Efficiency Analysis - {block_name}",
                                filename=f"efficiency_{block_name}.png",
                                ylabel="Efficiency (%)",
                                to_db=False
                            )
                    
                    # 生成摘要表格
                    summary_table = self.post_processor.create_summary_table(processed_df)
                    summary_path = os.path.join(self.output_dir, f"summary_{block_name}.csv")
                    summary_table.to_csv(summary_path, index=False)
                    print(f"    ✓ 摘要表格已保存: {summary_path}")
            
            # 其他测量数据生成表格而不是图
            other_cols = [col for col in df.columns 
                         if col not in ['freq'] + s_param_cols]
            
            if other_cols and 'HB' not in block_name.upper():
                # 生成原始数据表格
                table_path = os.path.join(self.output_dir, f"measurements_{block_name}.csv")
                df[['freq'] + other_cols].to_csv(table_path, index=False)
                print(f"    ✓ 测量数据表格已保存: {table_path}")
    
    def llm_suggest_parameters(self, design_spec: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """使用 LLM 建议参数"""
        print("\n步骤 4: LLM 参数建议")
        
        if not self.variables:
            print("⚠ 没有可调变量，跳过 LLM 建议")
            return None
        
        suggestions = self.llm.suggest_initial_parameters(design_spec, self.variables)
        
        if suggestions:
            print("✓ LLM 参数建议:")
            for name, value in suggestions.items():
                print(f"  {name}: {value:.6e}")
            
            # 保存建议
            suggestions_path = os.path.join(self.output_dir, "llm_suggestions.json")
            with open(suggestions_path, 'w', encoding='utf-8') as f:
                json.dump(suggestions, f, indent=2)
            print(f"✓ 建议已保存: {suggestions_path}")
        else:
            print("⚠ LLM 未能生成有效建议")
        
        return suggestions
    
    def run_optimized_simulation(self, parameters: Dict[str, float]) -> bool:
        """运行优化后的仿真"""
        print("\n步骤 5: 运行优化仿真")
        
        ds_path, results = self.engine.run_simulation(
            output_no=2,
            parameters=parameters
        )
        
        if results:
            print(f"✓ 优化仿真完成")
            print(f"  数据集: {ds_path}")
            
            # 保存结果
            results_path = os.path.join(self.output_dir, "optimized_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'ds_path': str(ds_path),
                    'blocks': list(results.keys()),
                    'parameters': parameters,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            print(f"✓ 优化结果已保存: {results_path}")
            
            # 绘制结果
            self._plot_results(results)
            
            return True
        else:
            print("✗ 优化仿真失败")
            return False
    
    def generate_reports(self):
        """生成报告和图表"""
        print("\n步骤 6: 生成报告和图表")
        
        # 创建摘要报告
        summary = {
            'workspace': self.workspace_path,
            'library': self.library_name,
            'design': self.design_name,
            'variables': len(self.variables),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_path = self.visualizer.create_summary_report(
            summary,
            filename="simulation_summary.txt"
        )
        
        # 如果有仿真数据，生成示例图表
        # 这里需要根据实际数据调整
        print("✓ 报告生成完成")
    
    def run_full_workflow(self, design_spec: Optional[Dict] = None):
        """运行完整的自动化流程"""
        print("\n" + "=" * 70)
        print("开始自动化设计流程")
        print("=" * 70)
        
        # 1. 连接
        if not self.connect():
            print("\n✗ 流程终止：连接失败")
            return False
        
        # 2. 加载设计
        if not self.load_design():
            print("\n✗ 流程终止：加载设计失败")
            return False
        
        # 3. 基线仿真
        baseline_ok = self.run_baseline_simulation()
        
        # 4. LLM 建议（如果提供了设计规格）
        if design_spec and self.variables:
            suggestions = self.llm_suggest_parameters(design_spec)
            
            # 5. 优化仿真
            if suggestions:
                self.run_optimized_simulation(suggestions)
        
        # 6. 生成报告
        self.generate_reports()
        
        print("\n" + "=" * 70)
        print("自动化流程完成！")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 70)
        
        return True
    
    def cleanup(self):
        """清理资源"""
        self.engine.close()


def main():
    """主函数 - 解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ADS 自动化仿真系统（命令行版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  1. 自动模式（使用当前 ADS 工作空间）:
     python main.py --auto
     
  2. 指定工作空间:
     python main.py --workspace "D:/ADS_Projects/MyProject" --library "MyLib" --design "PA_Design"
     
  3. 启用 LLM:
     python main.py --auto --use-llm --llm-url "http://localhost:11434/api/generate"
     
  4. 提供设计规格（JSON 文件）:
     python main.py --auto --design-spec design_spec.json
        """
    )
    
    parser.add_argument('--workspace', type=str, help='ADS 工作空间路径')
    parser.add_argument('--library', type=str, help='库名称')
    parser.add_argument('--design', type=str, help='设计名称')
    parser.add_argument('--output', type=str, default='./results', help='输出目录（默认: ./results）')
    parser.add_argument('--auto', action='store_true', help='自动检测当前 ADS 工作空间')
    parser.add_argument('--use-llm', action='store_true', help='启用 LLM 辅助')
    parser.add_argument('--llm-url', type=str, default='http://localhost:11434/api/generate', help='LLM API 地址')
    parser.add_argument('--llm-model', type=str, default='qwen2.5:14b', help='LLM 模型名称')
    parser.add_argument('--design-spec', type=str, help='设计规格 JSON 文件路径')
    
    args = parser.parse_args()
    
    # 确定工作空间参数
    if args.auto:
        # 自动检测（需要在 ADS 环境中运行）
        try:
            from keysight.ads import de
            if de.workspace_is_open():
                ws = de.active_workspace()
                workspace_path = str(ws.path) if hasattr(ws, 'path') else str(ws)
                
                # 获取第一个库和设计
                libs = list(ws.libraries)
                if libs:
                    library_name = libs[0].name
                    designs = list(libs[0].designs)
                    design_name = designs[0].name if designs else "default"
                else:
                    print("✗ 工作空间中没有库")
                    return 1
                
                print(f"✓ 自动检测到:")
                print(f"  工作空间: {workspace_path}")
                print(f"  库: {library_name}")
                print(f"  设计: {design_name}")
            else:
                print("✗ 没有打开的工作空间")
                return 1
        except ImportError:
            print("✗ 无法导入 ADS 模块，请在 ADS Python 环境中运行")
            return 1
    else:
        # 使用命令行参数
        if not all([args.workspace, args.library, args.design]):
            print("✗ 错误：必须提供 --workspace, --library, --design 或使用 --auto")
            parser.print_help()
            return 1
        
        workspace_path = args.workspace
        library_name = args.library
        design_name = args.design
    
    # 加载设计规格
    design_spec = None
    if args.design_spec and os.path.exists(args.design_spec):
        with open(args.design_spec, 'r', encoding='utf-8') as f:
            design_spec = json.load(f)
        print(f"✓ 已加载设计规格: {args.design_spec}")
    
    # LLM 配置
    llm_config = None
    if args.use_llm:
        llm_config = {
            'model_type': 'local',
            'api_url': args.llm_url,
            'model_name': args.llm_model,
            'verbose': True
        }
    
    # 创建自动化控制器
    automation = ADSAutomation(
        workspace_path=workspace_path,
        library_name=library_name,
        design_name=design_name,
        output_dir=args.output,
        use_llm=args.use_llm,
        llm_config=llm_config
    )
    
    try:
        # 运行工作流
        success = automation.run_full_workflow(design_spec)
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        automation.cleanup()


if __name__ == "__main__":
    sys.exit(main())
