"""
模块包初始化文件
使 adsapi 成为 Python 包
"""

__version__ = "1.0.0"
__author__ = "LLMRF Project"
__description__ = "ADS 自动化仿真系统 - 基于 LLM 的射频电路设计"

# 导入主要类和函数，方便外部调用
from .ads_engine import ADSEngine
from .netlist_parser import NetlistParser
from .visualizer import ResultVisualizer
from .llm_interface import LLMInterface, MockLLMInterface

__all__ = [
    'ADSEngine',
    'NetlistParser',
    'ResultVisualizer',
    'LLMInterface',
    'MockLLMInterface',
]

# 模块信息
print(f"ADS API 模块已加载 v{__version__}")
