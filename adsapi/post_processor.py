"""
后处理模块

用于处理 ADS 仿真结果，特别是大信号 HB 数据
支持自定义公式计算 PAE、功率等参数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
import re


class FormulaCalculator:
    """公式计算器 - 支持自定义公式"""
    
    def __init__(self):
        """初始化计算器"""
        self.formulas = {}
        self.results = {}
        
        # 注册默认的 PA 计算公式
        self._register_default_formulas()
    
    def _register_default_formulas(self):
        """注册默认的功放计算公式"""
        
        # 输出功率 (W)
        def calc_pout_w(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            P_out_W = 0.5 * Re(V_out[1] * conj(I_load[1]))
            
            需要的数据：
            - Vout[1]: 输出电压的基频分量
            - l_load.i[1]: 负载电流的基频分量
            """
            # 尝试不同的命名约定
            V_out_1 = data.get('Vout[1]', data.get('V_out[1]', data.get('Vload[1]', None)))
            I_load_1 = data.get('l_load.i[1]', data.get('I_load[1]', data.get('Iload[1]', None)))
            
            if V_out_1 is None:
                raise ValueError(f"缺少输出电压数据，可用的键: {list(data.keys())[:20]}")
            if I_load_1 is None:
                raise ValueError(f"缺少负载电流数据，可用的键: {list(data.keys())[:20]}")
            
            return 0.5 * np.real(V_out_1 * np.conj(I_load_1))
        
        # 输出功率 (dBm)
        def calc_pout_dbm(data: Dict[str, np.ndarray]) -> np.ndarray:
            """P_out_dBm = 10 * log10(P_out_W) + 30"""
            P_out_W = calc_pout_w(data)
            return 10 * np.log10(P_out_W + 1e-20) + 30
        
        # 直流功耗
        def calc_pdc_total(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            P_dc_total = Re(Vs_high1[0] * Is_high1.i[0] + Vs_high2[0] * Is_high2.i[0])
            
            需要的数据：
            - Vs_high1[0], Is_high1.i[0]: 第一路电源的直流分量
            - Vs_high2[0], Is_high2.i[0]: 第二路电源的直流分量
            """
            # 尝试不同的命名约定
            V_s1_0 = data.get('Vs_high1[0]', data.get('V_s_high1[0]', data.get('Vdc1[0]', None)))
            I_s1_0 = data.get('Is_high1.i[0]', data.get('I_s_high1[0]', data.get('Idc1[0]', None)))
            V_s2_0 = data.get('Vs_high2[0]', data.get('V_s_high2[0]', data.get('Vdc2[0]', None)))
            I_s2_0 = data.get('Is_high2.i[0]', data.get('I_s_high2[0]', data.get('Idc2[0]', None)))
            
            if V_s1_0 is None or I_s1_0 is None:
                raise ValueError(f"缺少电源1数据，可用的键: {list(data.keys())[:20]}")
            
            P_dc = np.real(V_s1_0 * np.conj(I_s1_0))
            
            # 如果有第二路电源
            if V_s2_0 is not None and I_s2_0 is not None:
                P_dc += np.real(V_s2_0 * np.conj(I_s2_0))
            
            return P_dc
        
        # 输入功率 (W)
        def calc_pin_w(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            P_in_W = 0.5 * Re(V_input[1] * conj(I_input[1]))
            
            需要的数据：
            - Vinput[1]: 输入电压的基频分量
            - I_input.i[1]: 输入电流的基频分量
            """
            V_in_1 = data.get('Vinput[1]', data.get('V_input[1]', data.get('Vin[1]', data.get('V_in[1]', None))))
            I_in_1 = data.get('I_input.i[1]', data.get('I_input[1]', data.get('Iin[1]', data.get('I_in[1]', None))))
            
            if V_in_1 is None:
                raise ValueError(f"缺少输入电压数据，可用的键: {list(data.keys())[:20]}")
            if I_in_1 is None:
                raise ValueError(f"缺少输入电流数据，可用的键: {list(data.keys())[:20]}")
            
            return 0.5 * np.real(V_in_1 * np.conj(I_in_1))
        
        # 输入功率 (dBm)
        def calc_pin_dbm(data: Dict[str, np.ndarray]) -> np.ndarray:
            """P_in_dBm = 10 * log10(P_in_W) + 30"""
            P_in_W = calc_pin_w(data)
            return 10 * np.log10(P_in_W + 1e-20) + 30
        
        # 功率增益 (dB)
        def calc_gain(data: Dict[str, np.ndarray]) -> np.ndarray:
            """Gain = P_out_dBm - P_in_dBm"""
            return calc_pout_dbm(data) - calc_pin_dbm(data)
        
        # PAE (功率附加效率)
        def calc_pae(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            PAE = 100 * (P_out_W - P_in_W) / P_dc_total
            
            单位：%
            """
            P_out = calc_pout_w(data)
            P_in = calc_pin_w(data)
            P_dc = calc_pdc_total(data)
            
            return 100 * (P_out - P_in) / (P_dc + 1e-20)
        
        # 效率
        def calc_efficiency(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            Efficiency = 100 * P_out_W / P_dc_total
            
            单位：%
            """
            P_out = calc_pout_w(data)
            P_dc = calc_pdc_total(data)
            
            return 100 * P_out / (P_dc + 1e-20)
        
        # 注册所有公式
        self.register_formula('P_out_W', calc_pout_w, 
                             description='输出功率 (W)')
        self.register_formula('P_out_dBm', calc_pout_dbm, 
                             description='输出功率 (dBm)')
        self.register_formula('P_dc_total', calc_pdc_total, 
                             description='直流功耗 (W)')
        self.register_formula('P_in_W', calc_pin_w, 
                             description='输入功率 (W)')
        self.register_formula('P_in_dBm', calc_pin_dbm, 
                             description='输入功率 (dBm)')
        self.register_formula('Gain', calc_gain, 
                             description='功率增益 (dB)')
        self.register_formula('PAE', calc_pae, 
                             description='功率附加效率 (%)')
        self.register_formula('Efficiency', calc_efficiency, 
                             description='效率 (%)')
    
    def register_formula(self, name: str, func: Callable, description: str = ""):
        """
        注册自定义公式
        
        Args:
            name: 公式名称
            func: 计算函数，接受 Dict[str, np.ndarray] 返回 np.ndarray
            description: 公式描述
        """
        self.formulas[name] = {
            'function': func,
            'description': description
        }
        print(f"✓ 注册公式: {name} - {description}")
    
    def list_formulas(self) -> List[str]:
        """列出所有已注册的公式"""
        return list(self.formulas.keys())
    
    def calculate(self, formula_name: str, data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        计算指定公式
        
        Args:
            formula_name: 公式名称
            data: 数据字典 {变量名: 数组}
            
        Returns:
            计算结果数组，失败返回 None
        """
        if formula_name not in self.formulas:
            print(f"✗ 未找到公式: {formula_name}")
            return None
        
        try:
            result = self.formulas[formula_name]['function'](data)
            self.results[formula_name] = result
            return result
        except Exception as e:
            print(f"✗ 计算公式 {formula_name} 失败: {e}")
            return None
    
    def calculate_all(self, data: Dict[str, np.ndarray], 
                     formulas: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        计算多个公式
        
        Args:
            data: 数据字典
            formulas: 要计算的公式列表，None表示全部
            
        Returns:
            结果字典 {公式名: 结果数组}
        """
        if formulas is None:
            formulas = self.list_formulas()
        
        results = {}
        for formula_name in formulas:
            result = self.calculate(formula_name, data)
            if result is not None:
                results[formula_name] = result
        
        return results


class PostProcessor:
    """后处理器 - 处理仿真结果"""
    
    def __init__(self, calculator: Optional[FormulaCalculator] = None):
        """
        初始化后处理器
        
        Args:
            calculator: 公式计算器，None则创建默认的
        """
        self.calculator = calculator if calculator else FormulaCalculator()
        print("ℹ 后处理器已初始化")
    
    def process_hb_data(self, hb_df: pd.DataFrame, 
                       formulas: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理 HB 仿真数据
        
        Args:
            hb_df: HB 数据的 DataFrame
            formulas: 要计算的公式列表，None表示全部
            
        Returns:
            包含计算结果的新 DataFrame
        """
        print("\n处理 HB 数据...")
        print(f"  原始列: {list(hb_df.columns)}")
        
        # 检查是否有 Mix 列（谐波次数标识）
        if 'Mix[1]' not in hb_df.columns and 'mix' not in [c.lower() for c in hb_df.columns]:
            print("  ⚠ 未找到谐波标识列（Mix），跳过后处理")
            return pd.DataFrame()
        
        # 提取不同谐波的数据
        # Mix[1] = 0 代表 DC 分量
        # Mix[1] = 1 代表基频分量  
        dc_data = hb_df[hb_df['Mix[1]'] == 0].copy()
        fund_data = hb_df[hb_df['Mix[1]'] == 1].copy()
        
        if dc_data.empty or fund_data.empty:
            print("  ⚠ 缺少 DC 或基频数据")
            return pd.DataFrame()
        
        print(f"  提取到 {len(dc_data)} 个 DC 数据点，{len(fund_data)} 个基频数据点")
        
        # 准备数据字典（合并DC和基频数据）
        data = {}
        
        # 添加扫描变量（如RFpower）
        if 'RFpower' in dc_data.columns:
            data['RFpower'] = dc_data['RFpower'].values
        
        # 遍历所有列，构建 var[0] 和 var[1] 格式的数据
        for col in hb_df.columns:
            if col in ['freq', 'Mix[1]', 'RFpower']:
                continue
            
            # DC 分量 [0]
            if col in dc_data.columns:
                data[f'{col}[0]'] = dc_data[col].values
            
            # 基频分量 [1]
            if col in fund_data.columns:
                data[f'{col}[1]'] = fund_data[col].values
        
        print(f"  准备的数据键: {list(data.keys())[:10]}...")  # 只显示前10个
        
        # 计算所有公式
        calc_results = self.calculator.calculate_all(data, formulas)
        
        if not calc_results:
            print("  ⚠ 没有计算出任何结果")
            return pd.DataFrame()
        
        # 创建新的 DataFrame
        result_df = pd.DataFrame()
        
        # 添加扫描变量
        if 'RFpower' in data:
            result_df['RFpower'] = data['RFpower']
        
        # 添加计算结果
        for name, values in calc_results.items():
            result_df[name] = values
            print(f"  ✓ 计算完成: {name}")
        
        return result_df
    
    def extract_harmonic_data(self, hb_df: pd.DataFrame, 
                             variable_names: List[str]) -> Dict[str, Dict[int, np.ndarray]]:
        """
        从 HB 数据中提取谐波分量
        
        Args:
            hb_df: HB 数据的 DataFrame
            variable_names: 要提取的变量名列表（不含[n]后缀）
            
        Returns:
            {变量名: {谐波次数: 数组}}
        """
        results = {}
        
        for var in variable_names:
            results[var] = {}
            
            # 查找所有谐波分量
            pattern = re.compile(rf'^{re.escape(var)}\[(\d+)\]$')
            
            for col in hb_df.columns:
                match = pattern.match(col)
                if match:
                    harmonic_num = int(match.group(1))
                    results[var][harmonic_num] = hb_df[col].values
        
        return results
    
    def create_summary_table(self, results_df: pd.DataFrame, 
                            metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建摘要表格
        
        Args:
            results_df: 计算结果的 DataFrame
            metrics: 要包含的指标列表
            
        Returns:
            摘要表格 DataFrame
        """
        if metrics is None:
            metrics = [col for col in results_df.columns if col != 'freq']
        
        summary = []
        for metric in metrics:
            if metric in results_df.columns:
                values = results_df[metric].values
                summary.append({
                    'Metric': metric,
                    'Min': f"{np.min(values):.4f}",
                    'Max': f"{np.max(values):.4f}",
                    'Mean': f"{np.mean(values):.4f}",
                    'Std': f"{np.std(values):.4f}"
                })
        
        return pd.DataFrame(summary)


if __name__ == '__main__':
    # 测试代码
    print("="*70)
    print("后处理模块测试")
    print("="*70)
    
    # 创建计算器
    calc = FormulaCalculator()
    
    print(f"\n可用公式: {calc.list_formulas()}")
    
    # 创建测试数据
    test_data = {
        'V_out[1]': np.array([10+2j, 12+3j, 15+4j]),
        'I_load[1]': np.array([0.5+0.1j, 0.6+0.15j, 0.7+0.2j]),
        'V_input[1]': np.array([1+0.2j, 1.2+0.25j, 1.5+0.3j]),
        'I_input[1]': np.array([0.05+0.01j, 0.06+0.015j, 0.07+0.02j]),
        'V_s_high1[0]': np.array([28.0, 28.0, 28.0]),
        'I_s_high1[0]': np.array([0.5, 0.6, 0.7]),
    }
    
    # 计算
    results = calc.calculate_all(test_data)
    
    print("\n计算结果:")
    for name, values in results.items():
        print(f"  {name}: {values}")
