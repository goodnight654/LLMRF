"""
滤波器自动设计模块

基于 Chebyshev 低通滤波器设计理论
支持批量生成不同参数的滤波器设计
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class ChebyshevFilterDesigner:
    """Chebyshev 低通滤波器设计器"""
    
    def __init__(self):
        """初始化设计器"""
        self.design_history = []
    
    def design_lpf_by_attenuation(self, 
                                  ripple_db: float,
                                  fc: float,
                                  fs: float,
                                  R0: float,
                                  La: float) -> Dict:
        """
        根据所需衰减设计低通滤波器
        
        Args:
            ripple_db: 通带波纹 (dB)
            fc: 通带截止频率 (Hz)
            fs: 阻带频率 (Hz)
            R0: 参考阻抗 (Ω)
            La: fs频率处所需的衰减 (dB)
            
        Returns:
            dict: 设计结果 {'L': [...], 'C': [...], 'N': int, 'Atten': float, 'gk': [...]}
        """
        ep = 10 ** (ripple_db / 10) - 1
        pi = np.pi
        wc = 2 * pi * fc  # 角频通带频率
        ws = 2 * pi * fs  # 角频阻带频率
        
        # 计算滤波器阶数
        N = round(np.sqrt(np.arccosh((10 ** (La / 10) - 1) / ep)) / np.arccosh(ws / wc)) - 1
        
        # 计算实际衰减
        Atten = -10 * np.log10(1 + ep * np.cosh((N * np.arccosh(ws / wc))) ** 2)
        Atten = round(Atten, 2)
        
        # 计算归一化元件值
        beta = np.log(1 / np.tanh(ripple_db / 17.37))
        gamma = np.sinh(beta / (2 * N))
        
        L = []
        C = []
        ak = []
        bk = []
        gk = []
        
        # 计算中间参数
        for k in range(1, N + 1):
            a1 = np.sin(((2 * k - 1) * pi) / (2 * N))
            ak.append(a1)
            
            b1 = gamma**2 + (np.sin(k * pi / N)) ** 2
            bk.append(b1)
        
        # 计算归一化g值
        for k in range(1, N + 1):
            if k == 1:
                gk.append(round(2 * ak[k - 1] / gamma, 4))
            else:
                gk.append(round((4 * ak[k - 2] * ak[k - 1]) / (bk[k - 2] * gk[k - 2]), 4))
            
            # 转换为实际元件值
            if k % 2 != 0:
                L.append(round(((R0 * gk[k - 1] / wc) / 1e-9), 2))  # nH
            else:
                C.append(round((gk[k - 1] / (R0 * wc)) / 1e-12, 2))  # pF
        
        design = {
            'L': L,
            'C': C,
            'N': N,
            'Atten': Atten,
            'gk': gk,
            'params': {
                'ripple_db': ripple_db,
                'fc': fc,
                'fs': fs,
                'R0': R0,
                'La_target': La,
                'La_actual': Atten
            }
        }
        
        self.design_history.append(design)
        return design
    
    def generate_netlist(self, design: Dict, design_name: str = "filter") -> str:
        """
        生成 ADS 网表
        
        Args:
            design: 设计结果字典
            design_name: 设计名称
            
        Returns:
            str: ADS 网表内容
        """
        L = design['L']
        C = design['C']
        N = design['N']
        
        netlist_lines = [
            f"! {design_name} - Chebyshev LPF Order {N}",
            f"! fc={design['params']['fc']/1e6:.1f}MHz, fs={design['params']['fs']/1e6:.1f}MHz",
            ""
        ]
        
        # 添加端口
        netlist_lines.append("PORT1 PORT1 1 0 Num=1 Z=50 Ohm")
        
        # 添加元件
        node = 1
        for i, l_val in enumerate(L):
            netlist_lines.append(f"L{i+1} L{i+1} {node} {node+1} L=L{i+1} nH")
            node += 1
        
        # 添加电容和地
        node = 1
        for i, c_val in enumerate(C):
            netlist_lines.append(f"C{i+1} C{i+1} {node+1} 0 C=C{i+1} pF")
            node += 1
        
        netlist_lines.append(f"PORT2 PORT2 {node} 0 Num=2 Z=50 Ohm")
        
        # 添加变量定义
        netlist_lines.append("")
        netlist_lines.append("VAR")
        for i, l_val in enumerate(L):
            netlist_lines.append(f"  L{i+1}={l_val}")
        for i, c_val in enumerate(C):
            netlist_lines.append(f"  C{i+1}={c_val}")
        netlist_lines.append("END VAR")
        
        return "\n".join(netlist_lines)
    
    def create_design_spec(self, design: Dict, design_name: str = "filter") -> Dict:
        """
        创建设计规格（用于ADS仿真）
        
        Args:
            design: 设计结果
            design_name: 设计名称
            
        Returns:
            dict: 设计规格字典
        """
        fc = design['params']['fc']
        fs = design['params']['fs']
        
        spec = {
            'name': design_name,
            'type': 'chebyshev_lpf',
            'order': design['N'],
            'components': {
                'inductors': {f'L{i+1}': val for i, val in enumerate(design['L'])},
                'capacitors': {f'C{i+1}': val for i, val in enumerate(design['C'])}
            },
            'simulation': {
                'type': 'S_Param',
                'freq_start': 0.01e9,  # 10 MHz
                'freq_stop': fs * 2,   # 2倍阻带频率
                'freq_step': 10e6      # 10 MHz
            },
            'specs': design['params']
        }
        
        return spec


class FilterDatasetGenerator:
    """滤波器数据集生成器"""
    
    def __init__(self, output_dir: str = "./filter_dataset"):
        """
        初始化数据集生成器
        
        Args:
            output_dir: 输出目录
        """
        self.designer = ChebyshevFilterDesigner()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✓ 数据集生成器已初始化")
        print(f"  输出目录: {output_dir}")
    
    def generate_random_specs(self, num_samples: int = 100) -> List[Dict]:
        """
        生成随机的滤波器规格
        
        Args:
            num_samples: 样本数量
            
        Returns:
            list: 规格列表
        """
        specs = []
        
        # 定义参数范围
        ripple_range = [0.01, 0.05, 0.1, 0.5, 1.0]  # dB
        fc_range = np.linspace(500e6, 2e9, 20)      # 500MHz - 2GHz
        R0 = 50  # 固定阻抗
        
        for i in range(num_samples):
            # 随机选择参数
            ripple_db = np.random.choice(ripple_range)
            fc = np.random.choice(fc_range)
            
            # fs 在 fc 的 1.5-3 倍之间
            fs_ratio = np.random.uniform(1.5, 3.0)
            fs = fc * fs_ratio
            
            # 所需衰减在 30-60 dB之间
            La = np.random.uniform(30, 60)
            
            specs.append({
                'id': f'filter_{i:04d}',
                'ripple_db': ripple_db,
                'fc': fc,
                'fs': fs,
                'R0': R0,
                'La': La
            })
        
        return specs
    
    def generate_grid_specs(self, 
                           ripple_values: List[float],
                           fc_values: List[float],
                           fs_ratios: List[float],
                           La_values: List[float],
                           R0: float = 50) -> List[Dict]:
        """
        生成网格化的规格（笛卡尔积）
        
        Args:
            ripple_values: 波纹值列表 (dB)
            fc_values: 截止频率列表 (Hz)
            fs_ratios: 阻带频率比例列表 (fs/fc)
            La_values: 衰减值列表 (dB)
            R0: 参考阻抗
            
        Returns:
            list: 规格列表
        """
        specs = []
        idx = 0
        
        for ripple in ripple_values:
            for fc in fc_values:
                for fs_ratio in fs_ratios:
                    for La in La_values:
                        fs = fc * fs_ratio
                        specs.append({
                            'id': f'filter_grid_{idx:04d}',
                            'ripple_db': ripple,
                            'fc': fc,
                            'fs': fs,
                            'R0': R0,
                            'La': La
                        })
                        idx += 1
        
        print(f"✓ 生成网格规格: {len(specs)} 个组合")
        return specs
    
    def design_batch(self, specs: List[Dict]) -> List[Dict]:
        """
        批量设计滤波器
        
        Args:
            specs: 规格列表
            
        Returns:
            list: 设计结果列表
        """
        designs = []
        
        print(f"\n开始批量设计 {len(specs)} 个滤波器...")
        
        for i, spec in enumerate(specs):
            try:
                design = self.designer.design_lpf_by_attenuation(
                    ripple_db=spec['ripple_db'],
                    fc=spec['fc'],
                    fs=spec['fs'],
                    R0=spec['R0'],
                    La=spec['La']
                )
                
                # 添加ID
                design['id'] = spec['id']
                designs.append(design)
                
                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{len(specs)}")
                    
            except Exception as e:
                print(f"  ✗ 设计失败 {spec['id']}: {e}")
        
        print(f"✓ 完成设计: {len(designs)}/{len(specs)}")
        return designs
    
    def save_dataset(self, designs: List[Dict], filename: str = "filter_dataset.json"):
        """
        保存数据集
        
        Args:
            designs: 设计列表
            filename: 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(designs, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据集已保存: {filepath}")
        print(f"  包含 {len(designs)} 个设计")
    
    def generate_netlists(self, designs: List[Dict]) -> Dict[str, str]:
        """
        为所有设计生成网表
        
        Args:
            designs: 设计列表
            
        Returns:
            dict: {设计ID: 网表内容}
        """
        netlists = {}
        netlist_dir = os.path.join(self.output_dir, "netlists")
        os.makedirs(netlist_dir, exist_ok=True)
        
        print(f"\n生成网表...")
        
        for design in designs:
            design_id = design['id']
            netlist = self.designer.generate_netlist(design, design_id)
            netlists[design_id] = netlist
            
            # 保存单个网表文件
            netlist_path = os.path.join(netlist_dir, f"{design_id}.net")
            with open(netlist_path, 'w', encoding='utf-8') as f:
                f.write(netlist)
        
        print(f"✓ 生成网表: {len(netlists)} 个")
        print(f"  保存位置: {netlist_dir}")
        
        return netlists


if __name__ == '__main__':
    # 测试代码
    print("="*70)
    print("Chebyshev 滤波器设计器测试")
    print("="*70)
    
    # 单个设计测试
    designer = ChebyshevFilterDesigner()
    
    design = designer.design_lpf_by_attenuation(
        ripple_db=0.1,
        fc=800e6,
        fs=1500e6,
        R0=50,
        La=40
    )
    
    print("\n设计结果:")
    print(f"  阶数: {design['N']}")
    print(f"  电感 (nH): {design['L']}")
    print(f"  电容 (pF): {design['C']}")
    print(f"  实际衰减: {design['Atten']} dB")
    
    # 网表生成测试
    netlist = designer.generate_netlist(design, "test_filter")
    print("\n生成的网表:")
    print(netlist)
    
    # 批量生成测试
    print("\n" + "="*70)
    print("批量数据集生成测试")
    print("="*70)
    
    generator = FilterDatasetGenerator("./test_filter_dataset")
    
    # 生成10个随机设计
    specs = generator.generate_random_specs(10)
    designs = generator.design_batch(specs)
    generator.save_dataset(designs, "test_dataset.json")
    generator.generate_netlists(designs)
