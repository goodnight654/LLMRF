import json
import requests
from typing import Dict, List, Optional, Any
import os


class LLMInterface:
    """LLM 接口类 - 支持本地和 API 调用"""
    
    def __init__(self, 
                 model_type: str = "local",
                 api_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "qwen3:14b",
                 api_key: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化 LLM 接口
        
        Args:
            model_type: 模型类型 ("local", "openai")
            api_url: API 地址
            model_name: 模型名称
            api_key: API 密钥
            verbose: 是否输出详细日志
        """
        self.model_type = model_type
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose
        
        # 系统提示词 - 射频PA设计专家
        # 重点强调单位规范和输出格式要求，确保LLM输出符合工程实际需求
        self.system_prompt = """# 射频功率放大器(PA)设计专家系统
## 专业领域
- 电路网表(Netlist)与SPICE模型(BSIM/MOSFET)
- S参数分析(S11/S21)、谐波平衡仿真、负载牵引
- PA核心指标：增益(dB)、功率附加效率(PAE%)、线性度(ACPR)、输出功率(dBm)
- 匹配网络设计(LC/微带线)、热管理、稳定性分析

## 输出规则（必须严格遵守）
1. **数值单位**：所有参数必须使用国际单位制(SI)
    - 电容：法拉(F) -> 1.5pF = 1.5e-12
    - 电感：亨利(H) -> 8nH = 8.0e-9
    - 电阻：欧姆(Ohm)
2. **JSON格式**：
   - 初始参数：{"C1": 1.5e-12, "L1": 8.0e-9} （纯数值，无单位字符串）
   - 优化策略：{"C1": "increase", "L1": "decrease"} （仅限"increase"/"decrease"）
3. **禁止行为**：
   - 输出注释/说明文字（JSON必须可直接解析）
    - 模糊表述（"适当调整" -> "C1从1.2e-12增至1.5e-12"）
    - 单位缺失或错误（1.5 != 1.5e-12）

## 示例
 正确：{"C1": 1.5e-12}
 错误：{"C1": "1.5pF"} / {"C1": 1.5} / {"C1": "increase slightly"}
"""
        
        self._log(f"LLM 接口已初始化: {model_type} - {model_name}")
    
    def _log(self, message: str, level: str = "INFO"):
        """内部日志"""
        if self.verbose:
            prefix = {"INFO": "INFO", "SUCCESS": "SUCCESS", "ERROR": "ERROR", "WARNING": "WARNING"}.get(level, "INFO")
            print(f"{prefix}: {message}")
    
    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        try:
            if self.model_type == "local":
                return self._call_local(prompt, temperature, max_tokens)
            elif self.model_type == "openai":
                return self._call_openai(prompt, temperature, max_tokens)
            else:
                self._log(f"不支持的模型类型: {self.model_type}", "ERROR")
                return None
        except Exception as e:
            self._log(f"LLM 调用失败: {e}", "ERROR")
            return None

    def _call_local(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        try:
            full_prompt = f"{self.system_prompt}\n\n用户问题：{prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            self._log(f"调用本地 LLM: {self.api_url}")
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                self._log("LLM 调用成功", "SUCCESS")
                return answer.strip()
            else:
                self._log(f"LLM 返回错误: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self._log(f"本地 LLM 调用失败: {e}", "ERROR")
            return None
    
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                self._log("LLM 调用成功", "SUCCESS")
                return answer.strip()
            else:
                self._log(f"LLM 返回错误: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self._log(f"OpenAI API 调用失败: {e}", "ERROR")
            return None
    
    def suggest_initial_parameters(self, 
                                  design_spec: Dict[str, Any],
                                  variables: List[Dict]) -> Optional[Dict[str, float]]:
        """建议初始参数（严格使用SI单位）
        """
        # 【关键优化】明确单位标准 + 数值格式约束
        prompt = f"""## 设计目标

## 【重要】单位规范
所有参数值必须使用国际单位制(SI)：

## 可调参数（当前值 -> 范围）
"""
        for var in variables:
            # 安全获取单位（兼容旧数据结构）
            unit = var.get('unit', 'SI')
            prompt += f"- {var['name']} ({var['param']}): {var['value']:.2e} {unit} -> [{var['min']:.2e}, {var['max']:.2e}] {unit}\n"
        
        prompt += """
## 输出要求
仅输出纯JSON（无任何注释/说明文字），格式：
{"参数名": 目标值（SI单位数值）}

示例：
{"C1": 1.5e-12, "L1": 8.0e-9}
"""
        
        self._log("请求 LLM 建议初始参数（SI单位制）...")
        response = self.call(prompt, temperature=0.3)
        
        if not response:
            return None
        
        # 增强JSON提取（处理可能的markdown代码块）
        try:
            # 处理 ```json ... ``` 包裹的情况
            if "```" in response:
                json_str = response.split("```")[1].strip()
                if json_str.startswith("json"):
                    json_str = json_str[4:].strip()
            else:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end] if start >= 0 and end > start else response
            
            suggestions = json.loads(json_str)
            # 验证数值类型
            for k, v in suggestions.items():
                if not isinstance(v, (int, float)):
                    self._log(f"参数 {k} 值非数值类型: {v}", "WARNING")
                    return None
            self._log(f"成功解析 {len(suggestions)} 个参数建议（SI单位）", "SUCCESS")
            return suggestions
        except json.JSONDecodeError as e:
            self._log(f"JSON解析失败: {e} | 原始响应: {response[:200]}", "ERROR")
            return None
        except Exception as e:
            self._log(f"参数建议解析异常: {e}", "ERROR")
            return None
    
    def suggest_optimization_strategy(self,
                                    current_results: Dict[str, float],
                                    target_results: Dict[str, float],
                                    variables: List[Dict]) -> Optional[Dict[str, str]]:
        """
        建议优化策略（严格限定方向词）
        """
        prompt = f"""## 性能对比（当前 -> 目标）

## 可调参数（当前值）
"""
        for var in variables:
            unit = var.get('unit', 'SI')
            prompt += f"- {var['name']} ({var['param']}): {var['value']:.2e} {unit}\n"
        
        prompt += """
## 输出要求
仅输出纯JSON（无任何额外文字），键为参数名，值严格限定为：

示例：
{"C1": "increase", "L1": "decrease"}
"""
        
        self._log("请求 LLM 建议优化策略...")
        response = self.call(prompt, temperature=0.2)  # 降低温度提升确定性
        
        if not response:
            return None
        
        try:
            if "```" in response:
                json_str = response.split("```")[1].strip()
                if json_str.startswith("json"):
                    json_str = json_str[4:].strip()
            else:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end] if start >= 0 and end > start else response
            
            strategy = json.loads(json_str)
            # 验证值有效性
            valid_vals = {"increase", "decrease"}
            for k, v in strategy.items():
                if v not in valid_vals:
                    self._log(f"策略 {k} 值无效: '{v}'（应为increase/decrease）", "WARNING")
                    return None
            self._log(f"成功解析优化策略（{len(strategy)}个参数）", "SUCCESS")
            return strategy
        except Exception as e:
            self._log(f"策略解析失败: {e} | 响应: {response[:150]}", "ERROR")
            return None
    
    def analyze_simulation_results(self, results: Dict[str, Any]) -> Optional[str]:
        """
        分析仿真结果（专业报告格式）
        """
        # 格式化结果（增强可读性）
        formatted = []
        for k, v in results.items():
            if k.endswith('_db'):
                formatted.append(f"{k.replace('_db','').upper()}: {v:.1f} dB")
            elif k == 'pae':
                formatted.append(f"PAE: {v:.1f}%")
            elif k == 'frequency':
                formatted.append(f"Frequency: {v/1e9:.2f} GHz")
            else:
                formatted.append(f"{k}: {v}")
        
        prompt = f"""## 仿真结果
{chr(10).join(formatted)}

## 分析要求
用专业射频工程师语言，分三部分：
1. 【优势】突出达标指标（带具体数值）
2. 【问题】指出关键差距（量化影响，如"增益缺口0.5dB导致输出功率不足"）
3. 【建议】给出可执行调整（参数名+调整方向+目标值，例："C1从1.2e-12F增至1.5e-12F"）

## 禁止
"""
        
        self._log("请求 LLM 分析仿真结果...")
        response = self.call(prompt, temperature=0.5, max_tokens=1500)
        
        if response:
            self._log("仿真分析完成", "SUCCESS")
        return response


class MockLLMInterface(LLMInterface):
    """用于测试的 Mock LLM 接口（不调用真实模型）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        self._log("Mock LLM 已启用，返回空响应", "WARNING")
        return None


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 70)
    llm = LLMInterface(
        model_type="local",
        model_name="qwen3:14b",
        verbose=True
    )

    # 测试1: 建议初始参数（含单位字段）
    print("\n【测试1】建议初始参数（SI单位制验证）")
    design_spec = {
        'type': 'PA',
        'frequency': 2.4e9,
        'target_gain': 15,
        'target_pae': 50
    }
    variables = [
        {'name': 'C1', 'param': 'C', 'value': 1.0e-12, 'min': 0.5e-12, 'max': 5e-12, 'unit': 'F'},
        {'name': 'L1', 'param': 'L', 'value': 10e-9, 'min': 5e-9, 'max': 20e-9, 'unit': 'H'},
        {'name': 'Rbias', 'param': 'R', 'value': 100, 'min': 50, 'max': 200, 'unit': 'Ohm'}
    ]
    suggestions = llm.suggest_initial_parameters(design_spec, variables)
    print(f"参数建议: {suggestions}")
    if suggestions and all(isinstance(v, (int, float)) for v in suggestions.values()):
        print("   验证：所有值均为数值类型（SI单位）")
    
    # 测试2: 优化策略
    print("\n【测试2】优化策略（方向词验证）")
    current = {'gain': 12.3, 'pae': 35.7}
    target = {'gain': 15.0, 'pae': 50.0}
    strategy = llm.suggest_optimization_strategy(current, target, variables)
    print(f"优化策略: {strategy}")
    if strategy and all(v in ["increase", "decrease"] for v in strategy.values()):
        print("   验证：所有策略值为有效方向词")
    
    # 测试3: 仿真分析
    print("\n【测试3】仿真结果分析")
    results = {
        'gain': 14.5, 
        'pae': 48.2, 
        's11_db': -15.3,
        'frequency': 2.4e9
    }
    analysis = llm.analyze_simulation_results(results)
    print(f"分析报告:\n{analysis}")
    print("=" * 70)