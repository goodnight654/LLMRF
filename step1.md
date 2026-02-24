
---

### 第四阶段：模型导出与部署 (Merge -> GGUF -> Ollama)

以下是训练完成后，将 LoRA 权重合并、转换为 GGUF 格式，并最终导入 Ollama 的一键式命令清单。

#### 1. 合并权重 (Merge LoRA)
使用 LLaMA-Factory 将你训练好的 LoRA 权重与 Qwen3-8B 基础模型合并，导出一个完整的 HuggingFace 模型。
在 `LLaMA-Factory` 目录下运行：

`powershell
llamafactory-cli export `
  --model_name_or_path G:\wenlong\models\Qwen3-8B `
  --adapter_name_or_path G:\wenlong\llmrf\LLaMA-Factory\saves\Qwen3-8B-Base\lora\train_q4_24g_safe `
  --template qwen3_nothink `
  --finetuning_type lora `
  --export_dir G:\wenlong\models\Qwen3-8B-RF-Merged `
  --export_size 2 `
  --export_device cpu
`
*(注：`export_device cpu` 是为了防止显存溢出，合并过程会在内存中进行，可能需要几分钟)*

#### 2. 转换为 GGUF 格式 (llama.cpp)
**llama.cpp 介绍**：这是一个纯 C/C++ 实现的推理引擎，专门用于在普通硬件（CPU/GPU）上高效运行大语言模型。它定义了 `.gguf` 文件格式，这是目前 Ollama 底层依赖的核心格式。

**步骤 2.1：克隆并安装 llama.cpp**
打开一个新的终端，在你喜欢的目录（例如 `G:\wenlong`）下运行：
`powershell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
`

**步骤 2.2：将 HuggingFace 模型转换为 GGUF (FP16)**
在 `llama.cpp` 目录下运行以下 Python 脚本，将刚才合并的模型转换为 16-bit 的 GGUF 格式：
`powershell
python convert_hf_to_gguf.py G:\wenlong\models\Qwen3-8B-RF-Merged --outfile G:\wenlong\models\Qwen3-8B-RF-Merged\qwen3-8b-rf-f16.gguf --outtype f16
`

**步骤 2.3：(可选但推荐) 量化为 4-bit GGUF**
为了让模型在 Ollama 中运行得更快且占用极低显存（约 5-6GB），建议将其量化为 `q4_k_m` 格式。
*(注意：Windows 下需要先编译 llama.cpp 才能得到 `llama-quantize.exe`。如果你不想编译，可以直接跳过这一步，Ollama 也支持直接导入 f16 格式，只是显存占用会大一些，约 16GB)*
`powershell
# 如果你编译了 llama.cpp，运行：
.\llama-quantize.exe G:\wenlong\models\Qwen3-8B-RF-Merged\qwen3-8b-rf-f16.gguf G:\wenlong\models\Qwen3-8B-RF-Merged\qwen3-8b-rf-q4_k_m.gguf q4_k_m
`

#### 3. 导入 Ollama
**步骤 3.1：创建 Modelfile**
在 `G:\wenlong\models\Qwen3-8B-RF-Merged` 目录下，新建一个名为 `Modelfile` 的无后缀文本文件，填入以下内容：
`	ext
FROM ./qwen3-8b-rf-f16.gguf
# 如果你做了量化，就把上面这行改成 FROM ./qwen3-8b-rf-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
`

**步骤 3.2：构建并运行 Ollama 模型**
在 `Modelfile` 所在的目录下打开终端，运行：
`powershell
ollama create qwen3-8b-rf -f Modelfile
ollama run qwen3-8b-rf
`
如果成功进入对话界面，说明你的微调模型已经成功部署！接下来就可以在 `llm_ads_loop_v2.py` 中把 `MODEL_NAME` 改为 `"qwen3-8b-rf"` 进行闭环测试了。
