"""
结果可视化模块

用于生成 ADS 仿真结果的图表
支持 S 参数、Smith 圆图等常见射频图表
无需 GUI，直接保存图片文件
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class ResultVisualizer:
    """仿真结果可视化器"""
    
    def __init__(self, output_dir: str = "./plots"):
        """
        初始化可视化器
        
        Args:
            output_dir: 图片输出目录
        """
        # 转换为绝对路径
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ 可视化器已初始化，输出目录: {self.output_dir}")
    
    def plot_s_parameters(self, 
                         freq: np.ndarray, 
                         s_params: Dict[str, np.ndarray],
                         title: str = "S Parameters",
                         filename: str = "s_parameters.png",
                         db_scale: bool = True) -> str:
        """
        绘制 S 参数曲线
        
        Args:
            freq: 频率数组 (Hz)
            s_params: S 参数字典，如 {'S11': array, 'S21': array}
            title: 图表标题
            filename: 输出文件名
            db_scale: 是否使用 dB 刻度
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        freq_ghz = freq / 1e9  # 转换为 GHz
        
        for param_name, values in s_params.items():
            if db_scale:
                # 转换为 dB
                values_db = 20 * np.log10(np.abs(values) + 1e-12)
                ax.plot(freq_ghz, values_db, label=param_name, linewidth=2)
            else:
                ax.plot(freq_ghz, np.abs(values), label=param_name, linewidth=2)
        
        ax.set_xlabel('Frequency (GHz)', fontsize=12)
        ax.set_ylabel('Magnitude (dB)' if db_scale else 'Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ S 参数图已保存: {filepath}")
        return filepath
    
    def plot_smith_chart(self,
                        s11: np.ndarray,
                        freq: Optional[np.ndarray] = None,
                        title: str = "Smith Chart",
                        filename: str = "smith_chart.png",
                        show_markers: bool = True) -> str:
        """
        绘制 Smith 圆图
        
        Args:
            s11: S11 复数数组
            freq: 频率数组（可选，用于标注）
            title: 图表标题
            filename: 输出文件名
            show_markers: 是否显示标记点
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # 将 S11 转换为反射系数
        gamma = s11
        
        # 转换为极坐标
        magnitude = np.abs(gamma)
        phase = np.angle(gamma)
        
        # 绘制轨迹
        ax.plot(phase, magnitude, 'b-', linewidth=2, label='S11 Trajectory')
        
        if show_markers and freq is not None:
            # 在起点和终点添加标记
            ax.plot(phase[0], magnitude[0], 'go', markersize=10, label='Start')
            ax.plot(phase[-1], magnitude[-1], 'ro', markersize=10, label='End')
        
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Smith 圆图已保存: {filepath}")
        return filepath
    
    def plot_pa_performance(self,
                          pin: np.ndarray,
                          pout: np.ndarray,
                          gain: Optional[np.ndarray] = None,
                          pae: Optional[np.ndarray] = None,
                          title: str = "PA Performance",
                          filename: str = "pa_performance.png") -> str:
        """
        绘制功率放大器性能曲线
        
        Args:
            pin: 输入功率 (dBm)
            pout: 输出功率 (dBm)
            gain: 增益 (dB)，可选
            pae: 功率附加效率 (%)，可选
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        num_plots = 1 + (gain is not None) + (pae is not None)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Pout vs Pin
        axes[plot_idx].plot(pin, pout, 'b-', linewidth=2, label='Pout')
        axes[plot_idx].plot(pin, pin, 'r--', linewidth=1, label='Linear Ref', alpha=0.5)
        axes[plot_idx].set_xlabel('Pin (dBm)', fontsize=12)
        axes[plot_idx].set_ylabel('Pout (dBm)', fontsize=12)
        axes[plot_idx].set_title('Output Power vs Input Power', fontsize=12)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Gain vs Pin
        if gain is not None:
            axes[plot_idx].plot(pin, gain, 'g-', linewidth=2)
            axes[plot_idx].set_xlabel('Pin (dBm)', fontsize=12)
            axes[plot_idx].set_ylabel('Gain (dB)', fontsize=12)
            axes[plot_idx].set_title('Gain vs Input Power', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # PAE vs Pin
        if pae is not None:
            axes[plot_idx].plot(pin, pae, 'm-', linewidth=2)
            axes[plot_idx].set_xlabel('Pin (dBm)', fontsize=12)
            axes[plot_idx].set_ylabel('PAE (%)', fontsize=12)
            axes[plot_idx].set_title('Power Added Efficiency vs Input Power', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ PA 性能图已保存: {filepath}")
        return filepath
    
    def plot_optimization_history(self,
                                 iterations: List[int],
                                 objective_values: List[float],
                                 title: str = "Optimization History",
                                 filename: str = "optimization_history.png",
                                 ylabel: str = "Objective Value") -> str:
        """
        绘制优化历史曲线
        
        Args:
            iterations: 迭代次数列表
            objective_values: 目标函数值列表
            title: 图表标题
            filename: 输出文件名
            ylabel: Y 轴标签
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(iterations, objective_values, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注最优点
        best_idx = np.argmin(objective_values)
        best_iter = iterations[best_idx]
        best_val = objective_values[best_idx]
        ax.plot(best_iter, best_val, 'r*', markersize=15, label=f'Best: {best_val:.4f}')
        ax.legend()
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 优化历史图已保存: {filepath}")
        return filepath
    
    def plot_parameter_sweep(self,
                           param_values: np.ndarray,
                           results: Dict[str, np.ndarray],
                           param_name: str = "Parameter",
                           title: str = "Parameter Sweep",
                           filename: str = "parameter_sweep.png") -> str:
        """
        绘制参数扫描结果
        
        Args:
            param_values: 参数值数组
            results: 结果字典，如 {'S11': array, 'S21': array}
            param_name: 参数名称
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for result_name, values in results.items():
            ax.plot(param_values, values, marker='o', linewidth=2, label=result_name)
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 参数扫描图已保存: {filepath}")
        return filepath
    
    def plot_from_dataframe(self, df, freq_col='freq', data_cols=None, 
                           title="Simulation Results", filename="results.png",
                           ylabel="Magnitude (dB)", to_db=True):
        """
        从 DataFrame 直接绘图（参考 4.py）
        
        Args:
            df: pandas DataFrame
            freq_col: 频率列名
            data_cols: 要绘制的数据列名列表
            title: 图表标题
            filename: 输出文件名
            ylabel: Y 轴标签
            to_db: 是否转换为 dB
        
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 提取频率（转换为 GHz）
        if freq_col in df.columns:
            freq = df[freq_col].values / 1e9
        else:
            freq = df.index.values / 1e9
        
        # 绘制每个数据列
        if data_cols is None:
            data_cols = [col for col in df.columns if col != freq_col]
        
        for col in data_cols:
            if col in df.columns:
                data = df[col].values
                
                # 转换为 dB（如果是复数）
                if to_db and np.iscomplexobj(data):
                    data_db = 20 * np.log10(np.abs(data) + 1e-12)
                    ax.plot(freq, data_db, label=col, linewidth=2)
                else:
                    ax.plot(freq, data, label=col, linewidth=2)
        
        ax.set_xlabel('Frequency (GHz)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # 保存
        filepath = os.path.join(self.output_dir, filename)
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 图表已保存: {filepath}")
        return filepath
    
    def create_summary_report(self,
                            results: Dict[str, Any],
                            filename: str = "summary_report.txt") -> str:
        """
        创建文本格式的摘要报告
        
        Args:
            results: 结果字典
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ADS 仿真结果摘要报告\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in results.items():
                f.write(f"{key}:\n")
                if isinstance(value, (int, float)):
                    f.write(f"  {value}\n")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                elif isinstance(value, list):
                    for item in value[:10]:  # 只显示前10项
                        f.write(f"  - {item}\n")
                    if len(value) > 10:
                        f.write(f"  ... (共 {len(value)} 项)\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"✓ 摘要报告已保存: {filepath}")
        return filepath


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("结果可视化模块测试")
    print("=" * 60)
    
    # 创建测试数据
    freq = np.linspace(1e9, 10e9, 100)  # 1-10 GHz
    s11 = 0.3 * np.exp(1j * np.linspace(0, 2*np.pi, 100))
    s21_mag = 10 - 5 * (freq / 1e10)  # dB
    s21 = 10 ** (s21_mag / 20)
    
    pin = np.linspace(-10, 20, 50)
    pout = pin + 15 - 0.1 * (pin + 5) ** 2
    gain = pout - pin
    pae = 50 * (1 - np.exp(-0.1 * (pin + 10)))
    
    # 创建可视化器
    viz = ResultVisualizer(output_dir="./test_plots")
    
    # 测试各种绘图功能
    viz.plot_s_parameters(
        freq, 
        {'S11': s11, 'S21': s21},
        title="测试 S 参数"
    )
    
    viz.plot_smith_chart(
        s11,
        freq,
        title="测试 Smith 圆图"
    )
    
    viz.plot_pa_performance(
        pin, pout, gain, pae,
        title="测试 PA 性能"
    )
    
    viz.plot_optimization_history(
        list(range(20)),
        [10 - 0.3*i + 0.1*np.random.randn() for i in range(20)],
        title="测试优化历史"
    )
    
    viz.create_summary_report({
        'Design': 'Test PA',
        'Frequency': '2.4 GHz',
        'Gain': 15.2,
        'PAE': 45.6
    })
    
    print("\n所有测试图表已生成完成！")
