"""
LLM 接口模块

用于连接大语言模型（本地或 API）
实现射频电路设计的智能辅助功能
支持参数建议、优化策略生成等

与 need.md 中的 LLM 项目对接
"""

import json
import requests
from typing import Dict, List, Optional, Any, Tuple
import os


class LLMInterface:
    """LLM 接口类 - 支持本地和 API 调用"""
    
    def __init__(self, 
                 model_type: str = "local",
                 api_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "qwen2.5:14b",
                 api_key: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化 LLM 接口
        
        Args:
            model_type: 模型类型 ("local" 或 "openai")
            api_url: API 地址（本地 Ollama 或 vLLM）
            model_name: 模型名称
            api_key: API 密钥（用于 OpenAI 等）
            verbose: 是否输出详细日志
        """
        self.model_type = model_type
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose
        
        # 系统提示词 - 射频电路设计专家
        self.system_prompt = """你是一个射频电路设计专家，特别擅长功率放大器（PA）设计。
你理解电路网表（Netlist）格式、SPICE 模型参数、S 参数、谐波平衡仿真等概念。
你可以根据设计目标（如增益、效率、带宽）给出优化建议和参数调整方向。

请用专业、简洁的语言回答，给出具体的数值建议。"""
        
        self._log(f"LLM 接口已初始化: {model_type} - {model_name}")
    
    def _log(self, message: str, level: str = "INFO"):
        """内部日志"""
        if self.verbose:
            prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}.get(level, "•")
            print(f"{prefix} {message}")
    
    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """
        调用 LLM
        
        Args:
            prompt: 用户提示词
            temperature: 温度参数（控制随机性）
            max_tokens: 最大生成长度
            
        Returns:
            str: LLM 的回复，失败返回 None
        """
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
        """
        调用本地 LLM（Ollama 或 vLLM）
        
        Ollama API 格式：
            POST /api/generate
            {
                "model": "qwen2.5:14b",
                "prompt": "...",
                "stream": false
            }
        """
        try:
            # 构建完整提示词
            full_prompt = f"{self.system_prompt}\n\n用户问题：{prompt}"
            
            # 准备请求
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
            
            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            
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
        """
        调用 OpenAI 兼容 API
        """
        try:
            headers = {
                "Content-Type": "application/json",
            }
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
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
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
        """
        根据设计指标建议初始参数
        
        Args:
            design_spec: 设计规格，如 {
                'type': 'PA',
                'frequency': 2.4e9,
                'target_gain': 15,
                'target_pae': 50
            }
            variables: 可调变量列表
            
        Returns:
            dict: 参数建议 {变量名: 建议值}
        """
        # 构建提示词
        prompt = f"""
设计目标：
{json.dumps(design_spec, indent=2, ensure_ascii=False)}

可调参数列表：
"""
        for var in variables:
            prompt += f"- {var['name']} ({var['param']}): 范围 [{var['min']}, {var['max']}], 当前 {var['value']}\n"
        
        prompt += """
请根据设计目标，建议每个参数的初始值。只需要给出参数名和建议值，格式如下：
{
    "C1": 1.5e-12,
    "L1": 8.0e-9,
    ...
}
"""
        
        self._log("请求 LLM 建议初始参数...")
        response = self.call(prompt, temperature=0.3)
        
        if not response:
            return None
        
        # 尝试从响应中提取 JSON
        try:
            # 查找 JSON 部分
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                suggestions = json.loads(json_str)
                self._log(f"成功解析参数建议: {len(suggestions)} 个", "SUCCESS")
                return suggestions
            else:
                self._log("无法从响应中提取 JSON", "WARNING")
                return None
        except Exception as e:
            self._log(f"解析 LLM 响应失败: {e}", "ERROR")
            return None
    
    def suggest_optimization_strategy(self,
                                    current_results: Dict[str, float],
                                    target_results: Dict[str, float],
                                    variables: List[Dict]) -> Optional[Dict[str, str]]:
        """
        根据当前结果和目标，建议优化策略
        
        Args:
            current_results: 当前性能，如 {'gain': 12, 'pae': 35}
            target_results: 目标性能，如 {'gain': 15, 'pae': 50}
            variables: 可调变量列表
            
        Returns:
            dict: 优化策略 {变量名: "增大" 或 "减小"}
        """
        prompt = f"""
当前性能：
{json.dumps(current_results, indent=2, ensure_ascii=False)}

目标性能：
{json.dumps(target_results, indent=2, ensure_ascii=False)}

可调参数：
"""
        for var in variables:
            prompt += f"- {var['name']} ({var['param']}): 当前 {var['value']}\n"
        
        prompt += """
请分析当前性能与目标的差距，建议每个参数应该增大还是减小。
格式：
{
    "C1": "increase",
    "L1": "decrease",
    ...
}
"""
        
        self._log("请求 LLM 建议优化策略...")
        response = self.call(prompt, temperature=0.3)
        
        if not response:
            return None
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                strategy = json.loads(json_str)
                self._log(f"成功解析优化策略", "SUCCESS")
                return strategy
            else:
                return None
        except Exception as e:
            self._log(f"解析优化策略失败: {e}", "ERROR")
            return None
    
    def analyze_simulation_results(self, results: Dict[str, Any]) -> Optional[str]:
        """
        分析仿真结果，给出专业评价
        
        Args:
            results: 仿真结果字典
            
        Returns:
            str: 分析报告
        """
        prompt = f"""
以下是 ADS 仿真结果：
{json.dumps(results, indent=2, ensure_ascii=False)}

请分析这些结果，指出：
1. 设计的优势
2. 可能存在的问题
3. 改进建议

请用简洁专业的语言回答。
"""
        
        self._log("请求 LLM 分析仿真结果...")
        response = self.call(prompt, temperature=0.5, max_tokens=1500)
        
        if response:
            self._log("分析完成", "SUCCESS")
        
        return response


class MockLLMInterface(LLMInterface):
    """
    Mock LLM 接口 - 用于测试
    不需要真实的 LLM 服务
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(model_type="mock", verbose=verbose)
        self._log("Mock LLM 接口已初始化（测试模式）")
    
    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """返回模拟响应"""
        self._log("Mock LLM 调用（返回模拟数据）")
        
        if "初始参数" in prompt or "initial" in prompt.lower():
            return """{
    "C1": 1.5e-12,
    "L1": 8.0e-9,
    "C2": 2.0e-12
}"""
        elif "优化策略" in prompt or "strategy" in prompt.lower():
            return """{
    "C1": "increase",
    "L1": "decrease"
}"""
        else:
            return "这是一个模拟的 LLM 响应。设计看起来合理，建议微调匹配网络参数以提高效率。"


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("LLM 接口模块测试")
    print("=" * 60)
    
    # 使用 Mock 接口进行测试
    llm = MockLLMInterface(verbose=True)
    
    # 测试 1: 建议初始参数
    print("\n测试 1: 建议初始参数")
    design_spec = {
        'type': 'PA',
        'frequency': 2.4e9,
        'target_gain': 15,
        'target_pae': 50
    }
    variables = [
        {'name': 'C1', 'param': 'C', 'value': 1.0e-12, 'min': 0.5e-12, 'max': 5e-12},
        {'name': 'L1', 'param': 'L', 'value': 10e-9, 'min': 5e-9, 'max': 20e-9}
    ]
    suggestions = llm.suggest_initial_parameters(design_spec, variables)
    print(f"参数建议: {suggestions}")
    
    # 测试 2: 建议优化策略
    print("\n测试 2: 建议优化策略")
    current = {'gain': 12, 'pae': 35}
    target = {'gain': 15, 'pae': 50}
    strategy = llm.suggest_optimization_strategy(current, target, variables)
    print(f"优化策略: {strategy}")
    
    # 测试 3: 分析结果
    print("\n测试 3: 分析仿真结果")
    results = {'gain': 14.5, 'pae': 48, 's11_db': -15}
    analysis = llm.analyze_simulation_results(results)
    print(f"分析报告:\n{analysis}")
    
    print("\n" + "=" * 60)
    print("提示：使用真实 LLM 需要启动 Ollama 或 vLLM 服务")
    print("例如：ollama run qwen2.5:14b")
    print("=" * 60)
