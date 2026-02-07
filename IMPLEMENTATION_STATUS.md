# LLMRF 项目进度与实施文档（根目录总览）

最后更新：2026-02-07

## 0. 一句话概览
你已经跑通了 **(1) 本地 LLM 推理服务（Ollama + Qwen3-14B）** 与 **(2) ADS Python API 自动生成滤波器原理图并完成仿真** 这两块关键基建；接下来要把它们串成“指标 → 设计 → 仿真 → 指标提取 → 反馈/优化 → 数据集/微调”的闭环，并把滤波器数据集做成可微调的 JSONL。

---

## 1. 当前项目目标（你这套系统要实现什么）

**最终目标**：输入 RF 设计指标（滤波器/PA），系统自动给出电路参数并用 ADS 仿真验证，形成可迭代优化闭环；随后用你生成的数据对模型做领域微调（先滤波器，再迁移到 PA）。

**闭环架构（现状 + 目标）**：

- 用户指标
- LLM Agent（本地 Ollama：Qwen3-14B）
- 参数生成 / 优化策略（PSO/贝叶斯/规则+LLM）
- ADS 自动化（生成网表/原理图、运行仿真）
- 结果解析与后处理（S 参数、HB：Pout/Gain/PAE…）
- 反馈给 LLM / 优化器

---

## 2. 你已经完成/验证的内容（里程碑）

### 2.1 本地大模型侧
-  已在 Ollama 部署 Qwen3-14B（本地推理服务已可用）
-  已具备通过 HTTP API 调用本地模型的接口代码雏形（见 `adsapi/llm_interface.py`）

### 2.2 ADS 自动化侧（滤波器）
- ✅ 在 `filter_design.py` 中验证：
  - 通过 ADS Python API 创建 workspace / library / schematic
  - 自动放置元件、端口、S 参数控制器
  - 生成网表并调用 CircuitSimulator 仿真
  - 读取 `.ds` 并绘图

- ✅ 在 `adsapi/` 中验证：
  - ADS 仿真引擎封装、网表变量解析、结果绘图、HB 后处理等模块已存在并可跑通基线

---

## 3. 代码模块地图（你现在的“可复用能力”在哪里）

### 3.1 根目录
- `run.py`：推荐入口。读取 `adsapi/config.json`，用 ADS 自带 Python 解释器运行 `adsapi/main.py`
- `filter_design.py`：端到端 demo（生成滤波器原理图 + 仿真 + 读 ds + 绘图）
- `need.md`：PA 自动化设计与微调总体设想（路线图/创新点）
- `PROJECT_GUIDE.md`：较详细的阶段性指南（偏“计划 + 使用说明”）

### 3.2 adsapi/（ADS 自动化主实现）
- `ads_engine.py`：连接工作空间/设计、生成网表、运行仿真、读 ds
- `main.py`：命令行主程序，串联 engine + parser + visualizer + post_processor
- `netlist_parser.py`：从网表提取可调变量（opt/tune 等）
- `visualizer.py`：S 参数等结果绘图
- `post_processor.py`：HB 后处理（Pout/Gain/PAE 等）
- `batch_filter_simulation.py`：滤波器批量设计 + ADS 仿真，用于生成数据集
- `config.json`：关键配置（ADS 安装路径/ADS Python 路径/LLM 配置/输出目录）

---

## 4. 当前缺口（下一阶段需要补齐的“连接件”）

1) **滤波器数据集产出规范**
- 目前已有批量仿真脚本，但“输出字段/目录结构/失败重试/日志”需要稳定为可长期生成数据的格式。

2) **LLM → 参数建议 → 仿真反馈** 的闭环尚未完成
- `LLMInterface` 已存在，但还没在滤波器数据集/优化循环中形成稳定接口与评估。

3) **微调数据（JSONL）尚未落盘**
- `PROJECT_GUIDE.md` 中提到 `llm_dataset_builder.py`，但当前仓库根目录/adsapi 中尚未提供实现。

4) **从滤波器迁移到 PA**
- PA 的 HB 仿真数据后处理模块已经有基础（`post_processor.py`），但还缺一个“PA 设计器 + 扫描器 + 数据集生成器”。

---

## 5. 推荐的“接下来每一步做什么”（按顺序执行）

> 目标：先把滤波器数据集稳定生成出来，再做指令数据（JSONL），再把 LLM 接入形成闭环。

### Step 1：统一环境与配置（一次性）
1. 打开并确认 `adsapi/config.json`：
   - `ads.install_path` 指向你的 ADS 安装目录
   - `ads.python_exe` 指向 ADS 自带 python.exe
2. 用 ADS Python 跑配置检查（如果你有 `adsapi/check_config.py`）：
   - 期望输出：能找到 ADS 安装路径、能 import keysight 包

### Step 2：验证 adsapi 的“基线仿真管线”（一次性）
使用根目录入口跑一次：
- `python run.py --workspace <你的workspace> --library <库名> --design <设计名>`

验收标准：
- 生成 `adsapi/results/` 或你配置的输出目录
- `design.net / variables.json / plots/` 等文件出现

### Step 3：跑通“小规模滤波器数据集”生成（建议先 3 个）
- 使用 `adsapi/batch_filter_simulation.py` 的 test 模式（如果脚本支持 `--mode test`）

验收标准：
- 生成 `filter_results/`（或你指定的 output_dir）
- 每个 design 目录里有 `.csv` 或 `.ds` 导出的 S 参数数据
- summary.csv 或 simulation_results.json 成功生成（若脚本已有）

### Step 4：固化“数据集 schema”（你后续训练会非常依赖）
建议固定为：
- `design_id`
- `spec`（ripple_db/fc/fs/La/R0 等）
- `circuit`（L/C 列表或网表参数）
- `metrics`（S11_max_dB、S21_passband_min_dB、S21_stopband_max_dB、passband_ripple_dB…）
- `artifacts`（csv/ds/netlist 路径）
- `status`（ok/failed + error message）

### Step 5：生成微调数据（JSONL，SFT/指令格式）
新增一个构建脚本（建议放 `adsapi/llm_dataset_builder.py`）：
- 输入：Step 4 的 dataset JSON/CSV
- 输出：JSONL（至少包含正向任务 + 反向任务）
  - 正向：指标 → 元件参数 + 预期性能
  - 反向：元件参数 → 性能预测
  - 可选：性能偏差 → 调参建议

### Step 6：把 LLM 接入滤波器闭环（先不用优化器也行）
先做最小闭环：
1. 给 LLM 一个规格，让它输出 L/C 初值
2. 调用 ADS 仿真
3. 提取指标后，把差距回馈给 LLM，让它给“增大/减小/范围调整”建议

验收标准：
- 形成可重复运行的迭代日志（第 1 轮→第 N 轮指标逐步逼近）

### Step 7：再引入优化器（PSO/贝叶斯/网格）
- LLM 负责“缩小搜索空间 + 给初值/方向”
- 优化器负责“在范围内找到最优”

### Step 8：迁移到 PA（HB 数据集生成 → 微调）
按 `need.md` 的路线：
- 先定一个可复现 PA 拓扑 + 变量表
- 做 HB 扫描生成 1000+ 条数据
- 用 `post_processor.py` 固化 Pout/Gain/PAE 计算与数据清洗
- 再做 PA 的 JSONL 指令数据与微调

---

## 6. 当前建议你优先回答/确认的 3 个现实问题（用于减少返工）

1) 你计划以哪个脚本作为“唯一入口”？
- 建议：保留 `run.py` 作为唯一入口，其他 demo 脚本只做验证。

2) 你的 ADS 版本是否统一为 ADS2025_Update1？
- 当前仓库里有 ADS2025_Update1 和 ADS2026 的混杂示例；建议以后以 `adsapi/config.json` 为准。

3) 滤波器数据集你希望以“真实仿真指标”为准，还是“解析网表 + 理论计算”为准？
- 建议以 ADS 仿真为准（对后续迁移 PA 更一致）。
