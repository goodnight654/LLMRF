"""
网表解析器模块

用于解析 ADS 网表文件，提取优化变量（opt/tune）
支持多种 ADS 网表格式
"""

import re
import os
from typing import Dict, List, Tuple, Optional


class NetlistParser:
    """ADS 网表解析器"""
    
    # 单位转换表
    UNIT_MULTIPLIERS = {
        'f': 1e-15,   # femto
        'p': 1e-12,   # pico
        'n': 1e-9,    # nano
        'u': 1e-6,    # micro
        'm': 1e-3,    # milli
        'k': 1e3,     # kilo
        'K': 1e3,
        'M': 1e6,     # mega
        'G': 1e9,     # giga
        'T': 1e12,    # tera
        '': 1.0
    }
    
    @staticmethod
    def parse_value_with_unit(value_str: str) -> float:
        """
        解析带单位的数值
        
        Examples:
            "1.37 pF" -> 1.37e-12
            "10 nH" -> 10e-9
            "50 Ohm" -> 50.0
            
        Args:
            value_str: 带单位的数值字符串
            
        Returns:
            float: 解析后的数值（SI 单位）
        """
        value_str = value_str.strip()
        
        # 匹配数值和单位前缀
        match = re.match(r'^([\d.eE+-]+)\s*([fpnumkKMGT])?\s*\w*$', value_str)
        if match:
            num = float(match.group(1))
            prefix = match.group(2) or ''
            multiplier = NetlistParser.UNIT_MULTIPLIERS.get(prefix, 1.0)
            return num * multiplier
        
        # 尝试直接转换数值
        try:
            return float(value_str.split()[0])
        except:
            return 0.0
    
    @staticmethod
    def extract_opt_tune_lines(netlist_content: str, display: bool = False) -> Tuple[List[str], List[str]]:
        """
        从网表文本中提取包含 opt 或 tune 的行
        
        Args:
            netlist_content: 网表文本内容
            display: 是否打印提取结果
            
        Returns:
            (lines, names): 匹配的行列表和参数名列表
        """
        # 匹配包含 tune{...} 或 opt{...} 的整行
        pattern_w = re.compile(r'^.*(?:tune|opt)\s*\{.*?\}.*$', re.MULTILINE | re.IGNORECASE)
        lines = pattern_w.findall(netlist_content)
        
        # 从每行提取参数名（格式：参数名=值）
        names = []
        for line in lines:
            # 查找 参数名=值 的模式
            match = re.search(r'([A-Za-z_]\w*)\s*=\s*[\d.eE+-]', line)
            if match:
                names.append(match.group(1))
        
        if display:
            print(f'找到 {len(names)} 个优化变量:')
            for i, name in enumerate(names, 1):
                print(f'  {i}. {name}')
            print(f'\n匹配的行示例（前 5 行）:')
            for line in lines[:5]:
                preview = line[:100] + '...' if len(line) > 100 else line
                print(f'  {preview}')
        
        return lines, names
    
    @staticmethod
    def parse_variables(netlist_content: str) -> List[Dict]:
        """
        从网表文本解析 opt/tune 变量的详细信息
        
        支持的格式:
            1. C:C1 ... C=1.37 pF tune{ 0.79 pF to 20 pF logScale }
            2. L:L1 ... L=10 nH tune{ 5 nH to 50 nH }
            3. R:R1 ... R=50 opt{ 10, 50, 100 }
            
        Args:
            netlist_content: 网表文本内容
            
        Returns:
            list: 变量信息字典列表，每个字典包含：
                - name: 变量名（如 C1, L1）
                - param: 参数类型（如 C, L, R）
                - value: 当前值
                - min: 最小值
                - max: 最大值
                - type: 优化类型（'tune' 或 'opt'）
                - device_type: 器件类型
        """
        variables = []
        seen_names = set()  # 避免重复
        
        lines = netlist_content.split('\n')
        
        for line in lines:
            if 'tune' not in line.lower() and 'opt' not in line.lower():
                continue
            
            # 提取器件名称：格式为 "Type:Name" 如 "C:C1" 或 "L:L1"
            device_match = re.match(r'^\s*([A-Za-z]+):([A-Za-z0-9_]+)', line)
            if not device_match:
                continue
            
            device_type = device_match.group(1)  # C, L, R 等
            device_name = device_match.group(2)  # C1, L1, R1 等
            
            if device_name in seen_names:
                continue
            
            # 模式1: tune{ min to max } 格式
            tune_pattern = re.compile(
                r'([A-Za-z])\s*=\s*([\d.eE+-]+)\s*[a-zA-Z]*\s+'
                r'tune\s*\{\s*([\d.eE+-]+)\s*[a-zA-Z]*\s+to\s+([\d.eE+-]+)\s*[a-zA-Z]*',
                re.IGNORECASE
            )
            
            tune_match = tune_pattern.search(line)
            if tune_match:
                param_type = tune_match.group(1)  # C 或 L
                cur_val = float(tune_match.group(2))
                min_val = float(tune_match.group(3))
                max_val = float(tune_match.group(4))
                
                seen_names.add(device_name)
                variables.append({
                    'name': device_name,
                    'param': param_type,
                    'value': cur_val,
                    'min': min_val,
                    'max': max_val,
                    'type': 'tune',
                    'device_type': device_type
                })
                continue
            
            # 模式2: opt{ min, current, max } 格式
            opt_pattern = re.compile(
                r'([A-Za-z])\s*=\s*'
                r'opt\s*\{\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\}',
                re.IGNORECASE
            )
            
            opt_match = opt_pattern.search(line)
            if opt_match:
                param_type = opt_match.group(1)
                min_val = float(opt_match.group(2))
                cur_val = float(opt_match.group(3))
                max_val = float(opt_match.group(4))
                
                seen_names.add(device_name)
                variables.append({
                    'name': device_name,
                    'param': param_type,
                    'value': cur_val,
                    'min': min_val,
                    'max': max_val,
                    'type': 'opt',
                    'device_type': device_type
                })
                continue
            
            # 模式3: tune{ min, current, max } opt{ min, current, max } 组合格式
            combined_pattern = re.compile(
                r'([A-Za-z])\s*=\s*'
                r'tune\s*\{\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\}\s*'
                r'opt\s*\{\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\}',
                re.IGNORECASE
            )
            
            combined_match = combined_pattern.search(line)
            if combined_match:
                param_type = combined_match.group(1)
                # 使用 opt 部分的值
                min_val = float(combined_match.group(5))
                cur_val = float(combined_match.group(6))
                max_val = float(combined_match.group(7))
                
                seen_names.add(device_name)
                variables.append({
                    'name': device_name,
                    'param': param_type,
                    'value': cur_val,
                    'min': min_val,
                    'max': max_val,
                    'type': 'opt',
                    'device_type': device_type
                })
        
        return variables
    
    @staticmethod
    def parse_from_file(netlist_path: str) -> List[Dict]:
        """
        从网表文件解析变量
        
        Args:
            netlist_path: 网表文件路径
            
        Returns:
            list: 变量信息列表
        """
        if not os.path.exists(netlist_path):
            print(f"✗ 网表文件不存在: {netlist_path}")
            return []
        
        try:
            with open(netlist_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return NetlistParser.parse_variables(content)
        except Exception as e:
            print(f"✗ 读取网表文件失败: {e}")
            return []
    
    @staticmethod
    def format_variable_table(variables: List[Dict]) -> str:
        """
        格式化变量表格用于打印
        
        Args:
            variables: 变量信息列表
            
        Returns:
            str: 格式化的表格字符串
        """
        if not variables:
            return "没有找到优化变量"
        
        lines = []
        lines.append("-" * 80)
        lines.append(f"{'变量名':<15} {'参数':<6} {'当前值':<15} {'最小值':<15} {'最大值':<15} {'类型':<6}")
        lines.append("-" * 80)
        
        for var in variables:
            lines.append(
                f"{var['name']:<15} "
                f"{var['param']:<6} "
                f"{var['value']:<15.6g} "
                f"{var['min']:<15.6g} "
                f"{var['max']:<15.6g} "
                f"{var['type']:<6}"
            )
        
        lines.append("-" * 80)
        lines.append(f"总计: {len(variables)} 个变量")
        
        return '\n'.join(lines)
    
    @staticmethod
    def variables_to_dict(variables: List[Dict]) -> Dict[str, Dict]:
        """
        将变量列表转换为字典格式（便于访问）
        
        Args:
            variables: 变量信息列表
            
        Returns:
            dict: {变量名: 变量信息}
        """
        return {var['name']: var for var in variables}


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("网表解析器测试")
    print("=" * 60)
    
    # 测试用例
    test_netlist = """
C:C1 ... C=1.37 pF tune{ 0.79 pF to 20 pF logScale }
L:L1 ... L=10 nH tune{ 5 nH to 50 nH }
R:R1 ... R=50 opt{ 10, 50, 100 }
C:C2 ... C=2.5 pF tune{ 1.0 pF to 10 pF }
    """
    
    print("\n测试网表内容:")
    print(test_netlist)
    
    print("\n提取 opt/tune 行:")
    lines, names = NetlistParser.extract_opt_tune_lines(test_netlist, display=True)
    
    print("\n解析变量:")
    variables = NetlistParser.parse_variables(test_netlist)
    print(NetlistParser.format_variable_table(variables))
    
    print("\n转换为字典:")
    var_dict = NetlistParser.variables_to_dict(variables)
    for name, info in var_dict.items():
        print(f"  {name}: {info}")
