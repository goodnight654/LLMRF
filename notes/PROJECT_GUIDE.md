# LLM+ADS 自动化设计系统 - 完整实施指南

## 项目概述

基于本地大语言模型（LLM）和 Keysight ADS 的射频电路自动化设计系统。

**目标**：从用户输入的性能指标，通过 LLM 推理生成电路参数，自动调用 ADS 仿真验证，形成完整的设计闭环。

**硬件配置**：RTX 4090 (24GB VRAM)  
**模型选择**：Qwen3-14B-Instruct (Int4/Int8 量化)

---

## 系统架构

```
用户输入指标
    ↓
LLM Agent (本地Qwen)
    ↓
参数生成 + 优化策略
    ↓
ADS Python API
    ↓
仿真引擎 (HB/SP)
    ↓
结果提取 + 后处理
    ↓
反馈给 LLM → 迭代优化
```

---

## 阶段一：环境搭建与滤波器数据集生成（当前阶段）

### 1.1 已完成的模块

 **ADS 自动化仿真接口** (`adsapi/`)
- `ads_engine.py` - ADS 仿真引擎封装
- `netlist_parser.py` - 网表解析
- `visualizer.py` - 结果可视化
- `post_processor.py` - 后处理（PAE, Gain, Efficiency 计算）

 **滤波器设计模块** (`filter_designer.py`)
- Chebyshev 低通滤波器自动设计
- 批量生成滤波器设计
- 网表自动生成

 **批量仿真工具** (`batch_filter_simulation.py`)
- 集成设计器 + ADS 仿真
- 自动提取 S 参数指标
- 支持大规模数据集生成

### 1.2 快速开始 - 生成第一个数据集

#### 步骤1：测试滤波器设计器

```bash
cd d:\Desktop\sjtu\LLMRF\adsapi
& "F:/Program Files (x86)/ADS2026/tools/python/python.exe" filter_designer.py
```

**预期输出**：
- 设计一个滤波器
- 显示电感、电容值
- 生成 ADS 网表
- 创建 10 个随机设计的数据集

#### 步骤2：运行小规模仿真测试（3个滤波器）

```bash
& "F:/Program Files (x86)/ADS2026/tools/python/python.exe" batch_filter_simulation.py --mode test
```

**这将**：
1. 创建 ADS 工作空间 `filter_test_wrk`
2. 设计 3 个不同规格的滤波器
3. 在 ADS 中创建原理图
4. 运行 S 参数仿真
5. 提取性能指标（S11, S21, 通带波纹）
6. 保存结果到 `filter_test_results/`

**生成的文件**：
```
filter_test_results/
├── designs/
│   └── filter_dataset.json          # 设计参数
├── simulations/
│   ├── test_800M/                    # 仿真数据
│   ├── test_1G/
│   └── test_1_5G/
├── simulation_results.json           # 完整结果
└── summary.csv                       # 性能摘要表
```

#### 步骤3：生成大规模数据集（108个滤波器）

```bash
& "F:/Program Files (x86)/ADS2026/tools/python/python.exe" batch_filter_simulation.py --mode dataset
```

**参数网格**：
- 通带波纹: [0.1, 0.5, 1.0] dB (3个)
- 截止频率: [800M, 1G, 1.5G, 2G] Hz (4个)
- 阻带比例: [1.5, 2.0, 2.5] (3个)
- 衰减要求: [30, 40, 50] dB (3个)
- **总计**: 3×4×3×3 = 108 个组合
### 1.3 数据集结构
生成的数据集包含：
```json
{
  "design_id": "filter_grid_0042",
  "design": {
    "L": [32.15, 41.23, 32.15],      // 电感值 (nH)
    "C": [8.92, 8.92],                // 电容值 (pF)
    "N": 5,                           // 滤波器阶数
    "params": {
      "ripple_db": 0.1,
      "fc": 1000000000.0,             // 截止频率 (Hz)
      "fs": 2000000000.0,             // 阻带频率 (Hz)
      "La_actual": 42.35              // 实际衰减 (dB)
    }
  },
  "metrics": {
    "S11_max_dB": -15.23,             // 通带最大回波损耗
    "S21_passband_min_dB": -0.52,     // 通带最小插损
    "S21_stopband_max_dB": -42.18,    // 阻带最大插损
    "passband_ripple_dB": 0.08        // 通带波纹
  }
}
```

---

## 阶段二：LLM 微调数据准备

### 2.1 构建指令微调数据集

将滤波器设计数据转换为 LLM 训练格式：

```python
# 创建 llm_dataset_builder.py
```

**数据格式**（JSONL）：

```json
{
  "instruction": "设计一个 Chebyshev 低通滤波器，要求：\n- 截止频率: 1 GHz\n- 阻带频率: 2 GHz\n- 阻带衰减: ≥40 dB\n- 通带波纹: ≤0.1 dB\n- 参考阻抗: 50Ω",
  "input": "",
  "output": "根据规格计算，需要 5 阶滤波器。元件参数如下：\n\n电感 (nH):\n- L1 = 32.15\n- L2 = 41.23\n- L3 = 32.15\n\n电容 (pF):\n- C1 = 8.92\n- C2 = 8.92\n\n预期性能：\n- 通带插损: -0.52 dB\n- 通带回波: -15.23 dB\n- 阻带衰减: -42.18 dB @ 2GHz"
}
```

**数据增强策略**：
1. **正向预测**：给定指标 → 预测元件值
2. **反向预测**：给定元件值 → 预测性能
3. **优化建议**：给定性能偏差 → 建议参数调整
4. **知识问答**：滤波器设计原理、权衡分析

### 2.2 创建数据构建脚本

```bash
cd adsapi
& "F:/Program Files (x86)/ADS2026/tools/python/python.exe" -m pip install --upgrade pip
```

创建 `llm_dataset_builder.py`:

```python
import json
import random

def build_instruction_dataset(simulation_results_path, output_path):
    """将仿真结果转换为 LLM 指令数据"""
    
    with open(simulation_results_path, 'r') as f:
        results = json.load(f)
    
    instructions = []
    
    for result in results:
        design = result['design']
        metrics = result['metrics']
        params = design['params']
        
        # 正向任务：设计滤波器
        instruction = {
            "instruction": f"设计一个 Chebyshev 低通滤波器，要求：\n"
                          f"- 截止频率: {params['fc']/1e6:.0f} MHz\n"
                          f"- 阻带频率: {params['fs']/1e6:.0f} MHz\n"
                          f"- 阻带衰减: ≥{params['La_target']:.0f} dB\n"
                          f"- 通带波纹: ≤{params['ripple_db']:.2f} dB\n"
                          f"- 参考阻抗: 50Ω",
            "input": "",
            "output": generate_design_output(design, metrics)
        }
        instructions.append(instruction)
        
        # 反向任务：性能预测
        reverse_inst = {
            "instruction": "预测以下滤波器的性能：",
            "input": format_circuit_params(design),
            "output": format_performance_prediction(metrics)
        }
        instructions.append(reverse_inst)
    
    # 保存为 JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for inst in instructions:
            f.write(json.dumps(inst, ensure_ascii=False) + '\n')
    
    print(f"✓ 生成 {len(instructions)} 条训练数据")
```

---

## 阶段三：LLM 本地部署与微调

### 3.1 部署推理引擎

#### 选项 A：Ollama（推荐，简单）

```bash
# Windows 下载安装 Ollama
# https://ollama.com/download

# 拉取模型
ollama pull qwen3:14b-instruct-q4_K_M

# 启动服务（默认 http://localhost:11434）
ollama serve
```

#### 选项 B：vLLM（性能最优）

```bash
pip install vllm

# 启动 API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --port 8000
```

### 3.2 测试 LLM 接口

修改 `llm_interface.py` 连接本地模型：

```python
llm = LLMInterface(
    model_type="local",
    api_url="http://localhost:11434/api/generate",  # Ollama
    model_name="qwen3:14b"
)

response = llm.generate_filter_design({
    'fc': 1e9,
    'fs': 2e9,
    'La': 40,
    'ripple_db': 0.1
})
```

### 3.3 微调模型（QLoRA）

使用 LLaMA-Factory 框架：

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 准备数据配置 dataset_info.json
{
  "filter_design": {
    "file_name": "filter_instructions.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}

# 启动 WebUI 微调
llamafactory-cli webui
```

**微调参数**（4090 推荐）：
- 模型：Qwen2.5-7B-Instruct
- 方法：QLoRA (Rank=64, Alpha=16)
- Batch Size：4
- Learning Rate：5e-5
- Epochs：3-5
- 预计时间：2-4小时

---

## 阶段四：PA 自动化设计扩展

### 4.1 PA 后处理已完成

 `post_processor.py` 已实现：
- 输出功率计算 (P_out_W, P_out_dBm)
- 直流功耗计算 (P_dc_total)
- 功率附加效率 (PAE)
- 增益 (Gain)

### 4.2 PA 设计流程

1. **定义 PA 拓扑**（Class-AB/Class-F）
2. **参数化匹配网络**
3. **HB 仿真扫描**（输入功率扫描）
4. **后处理计算 PAE, Gain**
5. **构建 PA 训练数据集**

### 4.3 创建 PA 设计器

```python
# pa_designer.py
class PADesigner:
    def design_class_ab_pa(self, 
                          freq: float,
                          pout_target: float,
                          pae_target: float):
        """设计 Class-AB PA"""
        
        # 1. 选择晶体管
        # 2. 计算偏置点
        # 3. 设计输入匹配
        # 4. 设计输出匹配
        # 5. 返回网表
        pass
```

---

## 阶段五：完整闭环集成

### 5.1 Agent 架构

```python
class RFDesignAgent:
    """RF 设计 Agent"""
    
    def __init__(self, llm, ads_engine, post_processor):
        self.llm = llm
        self.ads = ads_engine
        self.post = post_processor
    
    def design_with_feedback(self, specs, max_iterations=5):
        """迭代设计流程"""
        
        for i in range(max_iterations):
            # 1. LLM 生成设计
            design = self.llm.generate_design(specs)
            
            # 2. ADS 仿真
            sim_results = self.ads.run_simulation(design)
            
            # 3. 后处理
            metrics = self.post.calculate_metrics(sim_results)
            
            # 4. 检查是否满足规格
            if self.check_specs(metrics, specs):
                return design, metrics
            
            # 5. LLM 分析偏差，调整参数
            specs['feedback'] = f"当前性能: {metrics}, 需要改进..."
        
        return None
```

### 5.2 完整工作流

```
用户: "设计 2.4GHz PA, Pout=30dBm, PAE>50%"
  ↓
LLM: 生成初始设计（晶体管选型、匹配网络）
  ↓
ADS: HB 仿真
  ↓
后处理: PAE=45%, Pout=28dBm
  ↓
LLM: "输出功率不足，建议增加偏置电流，优化输出匹配"
  ↓
ADS: 重新仿真
  ↓
满足规格 → 输出设计
```

---

## 当前任务清单

###  已完成
1. ADS 自动化仿真接口
2. 滤波器设计器
3. 批量仿真工具
4. PA 后处理模块

### 🔄 进行中（接下来做）
1. **运行滤波器数据集生成**（本次任务）
   ```bash
   cd d:\Desktop\sjtu\LLMRF\adsapi
   & "F:/Program Files (x86)/ADS2026/tools/python/python.exe" batch_filter_simulation.py --mode test
   ```

2. **构建 LLM 训练数据**
   - 创建 `llm_dataset_builder.py`
   - 生成 JSONL 格式数据

3. **部署本地 LLM**
   - 安装 Ollama
   - 测试推理接口

### 📋 待完成
4. 微调 LLM（滤波器专家）
5. 集成 LLM + ADS 闭环
6. 扩展到 PA 设计
7. 论文撰写

---

## 常见问题

### Q1: 仿真速度慢怎么办？
A: 
- 减少频率点数（Step 从 0.01GHz 改为 0.05GHz）
- 使用并行仿真（分批运行）
- 使用更快的仿真器设置

### Q2: 如何验证数据集质量？
A:
- 检查 `summary.csv`，确保性能指标合理
- 可视化 S 参数曲线
- 对比设计目标和实际性能

### Q3: 内存不够怎么办？
A:
- 减小 batch_size（默认10，可改为5）
- 每批仿真后关闭 ADS 工作空间
- 使用 Int4 量化模型

---

## 下一步操作

立即执行：
```bash
cd d:\Desktop\sjtu\LLMRF\adsapi
& "F:/Program Files (x86)/ADS2026/tools/python/python.exe" batch_filter_simulation.py --mode test
```

这将生成第一个包含 3 个滤波器的测试数据集！
