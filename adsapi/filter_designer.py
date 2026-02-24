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
    
    def design_lpf_by_attenuation(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
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
        if order is None:
            N = round(np.sqrt(np.arccosh((10 ** (La / 10) - 1) / ep)) / np.arccosh(ws / wc)) - 1
        else:
            N = int(order)
        
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

    # --------------- 公用 g 值计算 ---------------
    def _compute_chebyshev_gk(self, ripple_db: float, N: int) -> List[float]:
        """计算 Chebyshev 归一化原型 g 值"""
        pi = np.pi
        beta = np.log(1 / np.tanh(ripple_db / 17.37))
        gamma = np.sinh(beta / (2 * N))
        ak, bk, gk = [], [], []
        for k in range(1, N + 1):
            ak.append(np.sin(((2 * k - 1) * pi) / (2 * N)))
            bk.append(gamma ** 2 + (np.sin(k * pi / N)) ** 2)
        for k in range(1, N + 1):
            if k == 1:
                gk.append(round(2 * ak[0] / gamma, 4))
            else:
                gk.append(round((4 * ak[k - 2] * ak[k - 1]) / (bk[k - 2] * gk[k - 2]), 4))
        return gk

    # --------------- 高通滤波器 (HPF) ---------------
    def design_hpf_by_attenuation(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
        """
        设计 Chebyshev 高通滤波器

        Args:
            ripple_db: 通带波纹 (dB)
            fc: 通带截止频率 (Hz), 高于 fc 为通带
            fs: 阻带频率 (Hz), 低于 fs 为阻带, fs < fc
            R0: 参考阻抗 (Ω)
            La: 阻带衰减目标 (dB, 正值)
            order: 指定阶数, None 则自动计算
        """
        assert fs < fc, f"HPF 要求 fs({fs}) < fc({fc})"

        ep = 10 ** (ripple_db / 10) - 1
        pi = np.pi
        wc = 2 * pi * fc
        omega_ratio = fc / fs  # 选择性比 >1

        if order is None:
            N_raw = np.arccosh(np.sqrt((10 ** (La / 10) - 1) / ep)) / np.arccosh(omega_ratio)
            N = max(2, int(np.ceil(N_raw)))
        else:
            N = int(order)

        Atten = -10 * np.log10(1 + ep * np.cosh(N * np.arccosh(omega_ratio)) ** 2)
        Atten = round(Atten, 2)

        gk = self._compute_chebyshev_gk(ripple_db, N)

        # HPF 元件转换: 奇数阶→串联电容, 偶数阶→并联电感
        C = []  # 串联电容 (pF)
        L = []  # 并联电感 (nH)
        for k in range(1, N + 1):
            if k % 2 != 0:
                c_val = 1.0 / (gk[k - 1] * R0 * wc)
                C.append(round(c_val / 1e-12, 2))
            else:
                l_val = R0 / (gk[k - 1] * wc)
                L.append(round(l_val / 1e-9, 2))

        design = {
            'L': L, 'C': C, 'N': N, 'Atten': Atten, 'gk': gk,
            'filter_band': 'highpass',
            'params': {
                'ripple_db': ripple_db, 'fc': fc, 'fs': fs,
                'R0': R0, 'La_target': La, 'La_actual': Atten
            }
        }
        self.design_history.append(design)
        return design

    # --------------- 带通滤波器 (BPF) ---------------
    def design_bpf_by_attenuation(
        self,
        ripple_db: float,
        f_center: float,
        bandwidth: float,
        fs_lower: float,
        fs_upper: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
        """
        设计 Chebyshev 带通滤波器

        Args:
            ripple_db: 通带波纹 (dB)
            f_center: 中心频率 (Hz)
            bandwidth: 通带带宽 (Hz)
            fs_lower: 下阻带频率 (Hz)
            fs_upper: 上阻带频率 (Hz)
            R0: 参考阻抗 (Ω)
            La: 阻带衰减目标 (dB, 正值)
            order: 指定阶数, None 则自动计算
        """
        f_lower = f_center - bandwidth / 2
        f_upper = f_center + bandwidth / 2
        f0 = np.sqrt(f_lower * f_upper)  # 几何中心频率
        w0 = 2 * np.pi * f0
        delta = bandwidth / f0  # 分数带宽

        ep = 10 ** (ripple_db / 10) - 1

        # 归一化阻带频率 (取更严格的一侧)
        omega_s_lo = abs(fs_lower / f0 - f0 / fs_lower) / delta
        omega_s_hi = abs(fs_upper / f0 - f0 / fs_upper) / delta
        omega_s = min(omega_s_lo, omega_s_hi)

        if order is None:
            N_raw = np.arccosh(np.sqrt((10 ** (La / 10) - 1) / ep)) / np.arccosh(omega_s)
            N = max(2, int(np.ceil(N_raw)))
        else:
            N = int(order)

        Atten = -10 * np.log10(1 + ep * np.cosh(N * np.arccosh(omega_s)) ** 2)
        Atten = round(Atten, 2)

        gk = self._compute_chebyshev_gk(ripple_db, N)

        # BPF 元件转换
        L_series, C_series = [], []  # 串联 LC 谐振器
        L_shunt, C_shunt = [], []    # 并联 LC 谐振器
        for k in range(1, N + 1):
            if k % 2 != 0:  # 串联元件 → 串联 LC
                Ls = gk[k - 1] * R0 / (delta * w0)
                Cs = delta / (gk[k - 1] * R0 * w0)
                L_series.append(round(Ls / 1e-9, 2))
                C_series.append(round(Cs / 1e-12, 2))
            else:            # 并联元件 → 并联 LC
                Cp = gk[k - 1] / (delta * R0 * w0)
                Lp = R0 * delta / (gk[k - 1] * w0)
                C_shunt.append(round(Cp / 1e-12, 2))
                L_shunt.append(round(Lp / 1e-9, 2))

        design = {
            'L_series': L_series, 'C_series': C_series,
            'L_shunt': L_shunt, 'C_shunt': C_shunt,
            'N': N, 'Atten': Atten, 'gk': gk,
            'filter_band': 'bandpass',
            'params': {
                'ripple_db': ripple_db,
                'f_center': f_center, 'bandwidth': bandwidth,
                'f_lower': f_lower, 'f_upper': f_upper,
                'fs_lower': fs_lower, 'fs_upper': fs_upper,
                'R0': R0, 'La_target': La, 'La_actual': Atten
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


class ButterworthFilterDesigner:
    """Butterworth 低通滤波器设计器"""

    def __init__(self):
        self.design_history = []

    def design_lpf_by_attenuation(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
        pi = np.pi
        wc = 2 * pi * fc
        ws = 2 * pi * fs

        # 计算滤波器阶数 (Butterworth)
        if order is None:
            N = int(
                np.ceil(
                    np.log10(10 ** (La / 10) - 1) / (2 * np.log10(ws / wc))
                )
            )
        else:
            N = int(order)

        # 归一化 g 值
        gk = [round(2 * np.sin((2 * k - 1) * pi / (2 * N)), 4) for k in range(1, N + 1)]

        # 转换为实际元件值
        L = []
        C = []
        for k in range(1, N + 1):
            if k % 2 != 0:
                L.append(round(((R0 * gk[k - 1] / wc) / 1e-9), 2))  # nH
            else:
                C.append(round((gk[k - 1] / (R0 * wc)) / 1e-12, 2))  # pF

        # 计算实际衰减
        Atten = -10 * np.log10(1 + (ws / wc) ** (2 * N))
        Atten = round(Atten, 2)

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

    # --------------- Butterworth HPF ---------------
    def design_hpf_by_attenuation(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
        """设计 Butterworth 高通滤波器 (HPF)"""
        assert fs < fc, f"HPF 要求 fs({fs}) < fc({fc})"
        pi = np.pi
        wc = 2 * pi * fc
        omega_ratio = fc / fs

        if order is None:
            N = int(np.ceil(np.log10(10 ** (La / 10) - 1) / (2 * np.log10(omega_ratio))))
        else:
            N = int(order)

        gk = [round(2 * np.sin((2 * k - 1) * pi / (2 * N)), 4) for k in range(1, N + 1)]

        C, L = [], []
        for k in range(1, N + 1):
            if k % 2 != 0:
                c_val = 1.0 / (gk[k - 1] * R0 * wc)
                C.append(round(c_val / 1e-12, 2))
            else:
                l_val = R0 / (gk[k - 1] * wc)
                L.append(round(l_val / 1e-9, 2))

        Atten = -10 * np.log10(1 + (omega_ratio) ** (2 * N))
        Atten = round(Atten, 2)

        design = {
            'L': L, 'C': C, 'N': N, 'Atten': Atten, 'gk': gk,
            'filter_band': 'highpass',
            'params': {
                'ripple_db': ripple_db, 'fc': fc, 'fs': fs,
                'R0': R0, 'La_target': La, 'La_actual': Atten
            }
        }
        self.design_history.append(design)
        return design

    # --------------- Butterworth BPF ---------------
    def design_bpf_by_attenuation(
        self,
        ripple_db: float,
        f_center: float,
        bandwidth: float,
        fs_lower: float,
        fs_upper: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
    ) -> Dict:
        """设计 Butterworth 带通滤波器 (BPF)"""
        f_lower = f_center - bandwidth / 2
        f_upper = f_center + bandwidth / 2
        f0 = np.sqrt(f_lower * f_upper)
        w0 = 2 * np.pi * f0
        delta = bandwidth / f0
        pi = np.pi

        omega_s_lo = abs(fs_lower / f0 - f0 / fs_lower) / delta
        omega_s_hi = abs(fs_upper / f0 - f0 / fs_upper) / delta
        omega_s = min(omega_s_lo, omega_s_hi)

        if order is None:
            N = int(np.ceil(np.log10(10 ** (La / 10) - 1) / (2 * np.log10(omega_s))))
        else:
            N = int(order)

        gk = [round(2 * np.sin((2 * k - 1) * pi / (2 * N)), 4) for k in range(1, N + 1)]

        L_series, C_series, L_shunt, C_shunt = [], [], [], []
        for k in range(1, N + 1):
            if k % 2 != 0:
                Ls = gk[k - 1] * R0 / (delta * w0)
                Cs = delta / (gk[k - 1] * R0 * w0)
                L_series.append(round(Ls / 1e-9, 2))
                C_series.append(round(Cs / 1e-12, 2))
            else:
                Cp = gk[k - 1] / (delta * R0 * w0)
                Lp = R0 * delta / (gk[k - 1] * w0)
                C_shunt.append(round(Cp / 1e-12, 2))
                L_shunt.append(round(Lp / 1e-9, 2))

        Atten = -10 * np.log10(1 + omega_s ** (2 * N))
        Atten = round(Atten, 2)

        design = {
            'L_series': L_series, 'C_series': C_series,
            'L_shunt': L_shunt, 'C_shunt': C_shunt,
            'N': N, 'Atten': Atten, 'gk': gk,
            'filter_band': 'bandpass',
            'params': {
                'ripple_db': ripple_db,
                'f_center': f_center, 'bandwidth': bandwidth,
                'f_lower': f_lower, 'f_upper': f_upper,
                'fs_lower': fs_lower, 'fs_upper': fs_upper,
                'R0': R0, 'La_target': La, 'La_actual': Atten
            }
        }
        self.design_history.append(design)
        return design


def get_filter_designer(filter_type: str):
    """按类型返回滤波器设计器"""
    ft = filter_type.lower().strip()
    if ft == "chebyshev":
        return ChebyshevFilterDesigner()
    if ft == "butterworth":
        return ButterworthFilterDesigner()
    if ft == "elliptic":
        raise ValueError("elliptic 暂未实现，请先使用 chebyshev 或 butterworth")
    raise ValueError(f"不支持的滤波器类型: {filter_type}")


class FilterDatasetGenerator:
    """滤波器数据集生成器"""
    
    def __init__(self, output_dir: str = "./filter_dataset", filter_type: str = "chebyshev"):
        """
        初始化数据集生成器
        
        Args:
            output_dir: 输出目录
        """
        self.filter_type = filter_type
        self.designer = get_filter_designer(filter_type)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✓ 数据集生成器已初始化")
        print(f"  输出目录: {output_dir}")
    
    def generate_random_specs(
        self,
        num_samples: int = 100,
        R0_values: Optional[List[float]] = None,
        order_values: Optional[List[int]] = None,
        filter_type: Optional[str] = None,
        filter_bands: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        生成随机的滤波器规格 (支持 LPF / HPF / BPF)
        
        Args:
            num_samples: 样本数量
            filter_bands: 要生成的频带类型列表, 默认 ['lowpass']
                          可选: ['lowpass', 'highpass', 'bandpass']
            
        Returns:
            list: 规格列表
        """
        specs = []
        filter_bands = filter_bands or ['lowpass']
        
        # 定义参数范围
        ripple_range = [0.01, 0.05, 0.1, 0.5, 1.0]  # dB
        fc_range = np.linspace(500e6, 2e9, 20)      # 500MHz - 2GHz
        R0_values = R0_values or [50]
        
        for i in range(num_samples):
            band = np.random.choice(filter_bands)
            ripple_db = float(np.random.choice(ripple_range))
            R0 = float(np.random.choice(R0_values))
            La = float(np.random.uniform(30, 60))
            ftype = filter_type or self.filter_type

            if band == 'highpass':
                fc = float(np.random.choice(fc_range))
                fs_ratio = np.random.uniform(0.3, 0.7)  # fs < fc
                fs = fc * fs_ratio
                spec = {
                    'id': f'filter_{i:04d}',
                    'ripple_db': ripple_db,
                    'fc': fc,
                    'fs': fs,
                    'R0': R0,
                    'La': La,
                    'filter_type': ftype,
                    'filter_band': 'highpass',
                }
            elif band == 'bandpass':
                f_center = float(np.random.choice(fc_range))
                frac_bw = np.random.uniform(0.05, 0.3)  # 分数带宽 5%-30%
                bandwidth = f_center * frac_bw
                fs_margin = np.random.uniform(1.3, 2.0)
                fs_lower = f_center - bandwidth / 2 * fs_margin
                fs_upper = f_center + bandwidth / 2 * fs_margin
                spec = {
                    'id': f'filter_{i:04d}',
                    'ripple_db': ripple_db,
                    'f_center': f_center,
                    'bandwidth': bandwidth,
                    'fs_lower': fs_lower,
                    'fs_upper': fs_upper,
                    'R0': R0,
                    'La': La,
                    'filter_type': ftype,
                    'filter_band': 'bandpass',
                }
            else:  # lowpass
                fc = float(np.random.choice(fc_range))
                fs_ratio = np.random.uniform(1.5, 3.0)
                fs = fc * fs_ratio
                spec = {
                    'id': f'filter_{i:04d}',
                    'ripple_db': ripple_db,
                    'fc': fc,
                    'fs': fs,
                    'R0': R0,
                    'La': La,
                    'filter_type': ftype,
                }

            if order_values:
                spec['order'] = int(np.random.choice(order_values))
            specs.append(spec)
        
        return specs
    
    def generate_grid_specs(
        self,
        ripple_values: List[float],
        fc_values: List[float],
        fs_ratios: List[float],
        La_values: List[float],
        R0_values: Optional[List[float]] = None,
        order_values: Optional[List[int]] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict]:
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
        
        R0_values = R0_values or [50]
        for ripple in ripple_values:
            for fc in fc_values:
                for fs_ratio in fs_ratios:
                    for La in La_values:
                        for R0 in R0_values:
                            fs = fc * fs_ratio
                            spec = {
                                'id': f'filter_grid_{idx:04d}',
                                'ripple_db': ripple,
                                'fc': fc,
                                'fs': fs,
                                'R0': R0,
                                'La': La,
                                'filter_type': (filter_type or self.filter_type)
                            }
                            if order_values:
                                spec['order'] = order_values[idx % len(order_values)]
                            specs.append(spec)
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
                filter_type = spec.get('filter_type', self.filter_type)
                self.designer = get_filter_designer(filter_type)
                filter_band = spec.get('filter_band', 'lowpass')

                if filter_band == 'highpass':
                    design = self.designer.design_hpf_by_attenuation(
                        ripple_db=spec['ripple_db'],
                        fc=spec['fc'],
                        fs=spec['fs'],
                        R0=spec['R0'],
                        La=spec['La'],
                        order=spec.get('order'),
                    )
                elif filter_band == 'bandpass':
                    design = self.designer.design_bpf_by_attenuation(
                        ripple_db=spec['ripple_db'],
                        f_center=spec['f_center'],
                        bandwidth=spec['bandwidth'],
                        fs_lower=spec['fs_lower'],
                        fs_upper=spec['fs_upper'],
                        R0=spec['R0'],
                        La=spec['La'],
                        order=spec.get('order'),
                    )
                else:  # lowpass (default)
                    design = self.designer.design_lpf_by_attenuation(
                        ripple_db=spec['ripple_db'],
                        fc=spec['fc'],
                        fs=spec['fs'],
                        R0=spec['R0'],
                        La=spec['La'],
                        order=spec.get('order'),
                    )
                
                # 添加ID
                design['id'] = spec['id']
                design['filter_type'] = filter_type
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
