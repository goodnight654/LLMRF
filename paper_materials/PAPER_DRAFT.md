# RF-FilterLLM: Automated RF Filter Design via Domain-Specific Fine-Tuning of Large Language Models

# RF-FilterLLM: 基于大语言模型领域微调的射频滤波器自动化设计

---

## Abstract / 摘要

### EN

We present RF-FilterLLM, a framework for automated radio-frequency (RF) filter design through domain-specific fine-tuning of large language models. We construct RF-Filter-SFT, a bilingual (Chinese–English) instruction-tuning dataset of 25,168 samples spanning Chebyshev/Butterworth lowpass, highpass, and bandpass filters. Using QLoRA (NF4 + LoRA $r{=}8$), we fine-tune Qwen3-8B on a single RTX 4090 with only 0.047% trainable parameters (3.83M/8.19B). The resulting Qwen3-8B-RF achieves filter-order prediction accuracy of 83.4% (vs. 19.5% baseline), 100% JSON compliance, and near-zero numerical error. A four-stage closed-loop reflective agent (Generate → Simulate → Evaluate → Reflect) further automates end-to-end design with Keysight ADS, converging in as few as 2 iterations. To the best of our knowledge, RF-FilterLLM is the first fine-tuned LLM specifically for RF filter synthesis, and the first to quantitatively benchmark filter-order prediction as an evaluation metric.

### 中文

本文提出 RF-FilterLLM，一种基于大语言模型领域微调的射频滤波器自动化设计框架。构建了包含 25,168 条中英双语样本的指令微调数据集 RF-Filter-SFT，覆盖 Chebyshev/Butterworth 低通、高通、带通滤波器。采用 QLoRA (NF4 + LoRA $r{=}8$) 在单张 RTX 4090 上微调 Qwen3-8B，仅需 0.047% 可训练参数 (3.83M/8.19B)。微调后的 Qwen3-8B-RF 将阶数预测准确率从 19.5% 提升至 83.4%，JSON 遵从率 100%，数值误差趋近于零。四步闭环反思 Agent（生成→仿真→评估→反思）实现与 Keysight ADS 的端到端自动化设计，最少 2 轮迭代收敛。据我们所知，RF-FilterLLM 是首个面向 RF 滤波器综合的微调 LLM，也是首次将阶数预测作为量化评估指标。

**Keywords / 关键词**: Large language model, RF filter design, QLoRA, instruction tuning, EDA automation, closed-loop agent / 大语言模型, 射频滤波器设计, QLoRA, 指令微调, EDA 自动化, 闭环 Agent

---

## I. Introduction / 引言

### EN

RF filters are essential building blocks in wireless systems—from cellular base stations to radar and satellite communications. Designing such filters requires selecting filter type, computing order $N$ from nonlinear relations among cutoff frequency $f_c$, stopband frequency $f_s$, passband ripple $L_r$, and stopband attenuation $L_A$, synthesizing prototype element values $g_k$, and iteratively validating in EDA tools. This process demands deep domain expertise and substantial engineering effort.

Recent works have applied LLMs to EDA. WiseEDA [1] leverages GPT-4 with prompt engineering for RF circuit design; ChipChat [2] and ChipGPT [3] target digital hardware (Verilog/VHDL). However, these approaches share three critical limitations:

1. **Domain knowledge gap** — General-purpose LLMs cannot internalize the nonlinear mapping from specifications to filter order (Eq. 1), resulting in near-random order predictions (19.5% accuracy).
2. **Proprietary API dependence** — Closed-source models raise privacy, cost, and latency concerns for defense and industrial RF applications.
3. **Single-band restriction** — Existing frameworks typically handle only one filter band (e.g., LPF), whereas practical systems require multi-band coverage.

**Contributions.** This paper makes four contributions:

- **(C1) RF-Filter-SFT Dataset**: The first large-scale bilingual instruction-tuning dataset for RF filter design (25,168 training samples, 3 bands, 2 types), with a novel *dynamic intent completion* mechanism that teaches the model to ask follow-up questions when critical parameters are missing.
- **(C2) Efficient Domain Adaptation**: QLoRA fine-tuning with 0.047% trainable parameters on consumer hardware (1× RTX 4090, 9.4 hours), improving order accuracy from 19.5% to 83.4%.
- **(C3) Multi-Band Unified Framework**: A single model unifies LPF/HPF/BPF design through classical frequency transformations.
- **(C4) Closed-Loop Reflective Agent**: A physics-informed iterative agent that reasons about S-parameter deviations (not blind search), converging in ≤5 iterations with ADS simulation.

### 中文

射频滤波器是无线通信系统的核心器件。其设计需要从 $f_c, f_s, L_r, L_A$ 的非线性关系中计算阶数 $N$、综合原型元件值 $g_k$、并在 EDA 工具中反复验证——高度依赖领域专业知识。

现有 LLM-EDA 工作（WiseEDA [1]、ChipChat [2]、ChipGPT [3]）存在三个关键局限：(1) **领域知识缺失**——通用 LLM 无法内化指标到阶数的非线性映射，阶数预测准确率仅 19.5%；(2) **依赖闭源 API**——隐私、成本、延迟不可控；(3) **单一频段**——仅支持单种滤波器。

**本文贡献：**
- **(C1)** 首个大规模中英双语 RF 滤波器指令微调数据集 RF-Filter-SFT（25,168 训练样本），引入动态意图补全机制。
- **(C2)** 仅 0.047% 参数的 QLoRA 微调，阶数准确率 19.5% → 83.4%，单张 4090 训练 9.4 小时。
- **(C3)** 单模型经频率变换统一 LPF/HPF/BPF 设计。
- **(C4)** 基于物理语义推理的闭环反思 Agent，与 ADS 仿真集成，最少 2 轮收敛。

---

## II. Related Work / 相关工作

### EN

**LLMs for EDA.** ChipChat [2] first explored conversational hardware design; ChipGPT [3] proposed LLM-driven automated hardware flows. WiseEDA [1] demonstrated GPT-4-based RF circuit design with PSO optimization. All rely on prompt engineering of closed-source models without domain-specific fine-tuning.

**Parameter-Efficient Fine-Tuning.** LoRA [4] injects low-rank adapters into pretrained weights; QLoRA [5] further reduces memory 3–4× via NF4 quantization. We adopt QLoRA to fine-tune 8B-scale models on consumer GPUs.

**RF Filter Design.** Classical synthesis follows Matthaei–Young–Jones theory [6], where filter order $N$ is determined by a nonlinear function of specification parameters (Eq. 1). This mathematical relationship is the core challenge for LLM-based design.

**Reflective Reasoning.** Reflexion [8] introduced verbal reinforcement learning for iterative self-correction. We adapt this paradigm to RF design, where the LLM diagnoses simulation failures using S-parameter semantics.

### 中文

**LLM-EDA**：ChipChat [2]、ChipGPT [3] 面向数字硬件；WiseEDA [1] 基于 GPT-4 + PSO 的 RF 电路设计。均依赖闭源模型提示工程，未做领域微调。

**参数高效微调**：LoRA [4] 注入低秩适配器；QLoRA [5] 通过 NF4 量化进一步降低 3–4× 显存。本文采用 QLoRA 在消费级 GPU 上微调 8B 模型。

**RF 滤波器设计**：经典综合遵循 Matthaei–Young–Jones 理论 [6]，阶数 $N$ 由指标参数的非线性函数确定（公式 1），是 LLM 面临的核心挑战。

**反思推理**：Reflexion [8] 提出的言语强化学习用于迭代自校正。本文将此范式适配至 RF 设计领域。

---

## III. Methodology / 方法

### III-A. RF-Filter-SFT Dataset / 数据集构建

#### EN

**Parameter space.** Type $\mathcal{T} \in \{\text{chebyshev}, \text{butterworth}\}$, band $\mathcal{B} \in \{\text{LPF}, \text{HPF}, \text{BPF}\}$, with $f_c \in [100\text{M}, 3\text{G}]$ Hz, $k_s = f_s/f_c \in [1.2, 3.0]$, $L_r \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$ dB, $L_A \in [20, 60]$ dB, $R_0 \in \{50, 75, 100\}$ Ω. BPF adds $f_0, \Delta f, f_{s,\text{lower}}, f_{s,\text{upper}}$.

**Order computation.** For Chebyshev filters:

$$N = \left\lceil \frac{\cosh^{-1}\!\left(\sqrt{(10^{L_A/10}-1)/(10^{L_r/10}-1)}\right)}{\cosh^{-1}(k_s)} \right\rceil \tag{1}$$

**Data format.** Each sample is a ShareGPT multi-turn conversation with a system prompt enforcing strict JSON output. The model must output all design parameters including order $N$ (inferred, not given in input).

**Dynamic intent completion (Innovation C1).** The dataset implements three interaction modes:

| Mode | Ratio | Description |
|:---:|:---:|:---|
| Full | 86.8% | Complete specs → direct JSON output |
| Followup-Q | 3.9% | Missing critical param → model asks follow-up |
| Followup-R | 9.3% | User provides missing param → model completes |

This teaches the model to distinguish *must-ask parameters* (e.g., $L_A$ — without which $N$ cannot be computed) from *auto-computable parameters* (e.g., $N$ — derived from Eq. 1).

**Bilingual mixing.** 82.6% Chinese, 17.4% English, including mixed-language queries (e.g., "帮我设计 chebyshev LPF").

**Table I: RF-Filter-SFT Statistics / 数据集统计**

| Split | Samples | LPF | HPF | BPF | ZH:EN |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Train | 25,168 | 40.3% | 25.7% | 30.1% | 82.6:17.4 |
| Val | 1,415 | 37.8% | 28.9% | 30.1% | 82.5:17.5 |
| Test | 1,372 | 41.9% | 29.4% | 24.7% | 83.3:16.7 |
| **Total** | **27,955** | | | | |

#### 中文

**参数空间**：类型 $\mathcal{T} \in \{\text{chebyshev}, \text{butterworth}\}$，频段 $\mathcal{B} \in \{\text{LPF}, \text{HPF}, \text{BPF}\}$，$f_c \in [100\text{M}, 3\text{G}]$ Hz, $k_s \in [1.2, 3.0]$, $L_r \in \{0.01, ..., 1.0\}$ dB, $L_A \in [20, 60]$ dB, $R_0 \in \{50, 75, 100\}$ Ω。BPF 额外定义 $f_0, \Delta f$ 及上下阻带频率。

**阶数计算**：公式 (1)。**数据格式**：ShareGPT 多轮对话，系统提示词强制 JSON 输出。

**动态意图补全（创新 C1）**：三种交互模式——Full (86.8%), Followup-Q (3.9%), Followup-R (9.3%)。训练模型区分"必须追问参数"与"可自动推导参数"。

**双语混合**：中文 82.6%，英文 17.4%，支持混合查询。统计见表 I。

---

### III-B. QLoRA Fine-Tuning / QLoRA 微调

#### EN

We fine-tune Qwen3-8B (8.19B params, 36 layers, 128 heads) with QLoRA [5]. The weight update:

$$h = \left(W_0 + \frac{\alpha}{r} B A\right) x \tag{2}$$

where $W_0 \in \mathbb{R}^{d \times d}$ is frozen, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, only $A, B$ trainable.

**Table II: Training Configuration / 训练配置**

| Parameter | Value |
|:---|:---:|
| Base model | Qwen3-8B (8.19B) |
| Quantization | NF4 (4-bit NormalFloat) |
| LoRA rank $r$ / scaling $\alpha$ | 8 / 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Trainable params | 3.83M (0.047%) |
| Learning rate | $1 \times 10^{-4}$ (cosine, 5% warmup) |
| Effective batch size | 16 (1 × 16 grad accum) |
| Max sequence length | 2,048 tokens |
| Epochs / Steps | 2 / 3,146 |
| Final training loss | 0.2785 |
| Wall time | 9h 24min (1× RTX 4090 24GB) |
| Framework / Template | LLaMA-Factory [9] / qwen3_nothink |

**Training dynamics** (Fig. 5, `training_loss_curve.pdf`): Loss drops from 1.52 to ~0.4 in the first 200 steps (JSON format learning), then gradually decreases through Epoch 1 (order reasoning acquisition), stabilizing at 0.28 in Epoch 2.

#### 中文

采用 QLoRA [5] 微调 Qwen3-8B，权重更新如公式 (2)。训练配置见表 II。损失曲线（图 5）：前 200 步从 1.52 快速降至 ~0.4（格式学习阶段），Epoch 1 期间缓慢下降（阶数推理习得），Epoch 2 稳定在 0.28。

---

### III-C. Multi-Band Filter Synthesis / 多频段滤波器综合

#### EN

The framework unifies three filter bands through frequency transformations of the lowpass prototype (Innovation C3):

**Lowpass (LPF)** — impedance/frequency denormalization:

$$L_k = \frac{g_k R_0}{2\pi f_c}, \quad C_k = \frac{g_k}{2\pi f_c R_0} \tag{3}$$

**Highpass (HPF)** — frequency inversion ($L \leftrightarrow C$):

$$C_{\text{HPF}} = \frac{1}{g_k R_0 \omega_c}, \quad L_{\text{HPF}} = \frac{R_0}{g_k \omega_c} \tag{4}$$

**Bandpass (BPF)** — narrowband transformation with fractional bandwidth $\delta = \Delta f / f_0$:

$$L_s = \frac{g_k R_0}{\delta \omega_0},\ C_s = \frac{\delta}{g_k R_0 \omega_0},\ C_p = \frac{g_k}{\delta R_0 \omega_0},\ L_p = \frac{R_0 \delta}{g_k \omega_0} \tag{5}$$

where $(L_s, C_s)$ form series resonators and $(L_p, C_p)$ form shunt resonators. BPF design involves a **5D parameter space** $(f_0, \Delta f, f_{s,\text{lower}}, f_{s,\text{upper}}, L_A)$ vs. 2D for LPF/HPF, explaining its higher prediction difficulty.

#### 中文

通过低通原型的频率变换统一三种频段（创新 C3）：LPF 阻抗/频率去归一化 (3)；HPF 频率倒置 (4)——电感电容互换；BPF 窄带变换 (5)——每个原型元件变为串联或并联谐振对。BPF 涉及 **5 维**参数空间（vs. LPF/HPF 的 2 维），解释了其预测难度更高。

---

### III-D. Closed-Loop Reflective Agent / 闭环反思 Agent

#### EN

We design a four-stage autonomous agent for end-to-end filter design (Innovation C4):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Algorithm 1: RF-FilterLLM Closed-Loop Filter Design
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input:  User specification S = {type, band, f_c, f_s,
            L_r, L_A, R_0, [f_0, Δf, f_s_lower, f_s_upper]}
          Fine-tuned model M = Qwen3-8B-RF
          Max iterations T = 5
          Pass criteria E = {S11 < −10 dB, |ripple| ≤ L_r,
                             atten ≥ L_A}
  Output: Validated design parameters P* with S-param data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1: P ← M.generate(system_prompt, S)       ▷ JSON output
   2: if P has missing critical params then
   3:     Q ← M.ask_followup(S)              ▷ Intent completion
   4:     S ← S ∪ user_response(Q)
   5:     P ← M.generate(system_prompt, S)
   6: end if
   7: for t = 1 to T do
   8:     netlist ← synthesize_netlist(P)     ▷ Eqs. (3)-(5)
   9:     results ← ADS.simulate(netlist)     ▷ hpeesofsim API
  10:     metrics ← extract(results)          ▷ S11, S21, ripple
  11:     if evaluate(metrics, E) = PASS then
  12:         return P, results               ▷ Design validated ✓
  13:     end if
  14:     feedback ← format_diagnosis(metrics, E, S)
  15:     P ← M.reflect(P, feedback)          ▷ Physics reasoning
  16: end for
  17: return P, results                       ▷ Best effort
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Key innovation (Line 15):** Unlike GA/PSO blind optimization, the LLM performs *physics-aware causal reasoning*. Example reflections from actual ADS runs:
- "S11 = −9.6 dB > −10 dB → passband matching insufficient → reduce ripple_db from 0.5 to 0.25"
- "Stopband attenuation 25 dB < target 40 dB → increase order from 3 to 5"

#### 中文

设计四步自主 Agent 实现端到端滤波器设计（创新 C4），见 Algorithm 1。

**核心创新（第 15 行）**：区别于 GA/PSO 盲目搜索，LLM 执行**物理语义因果推理**。实际 ADS 运行中的反思示例：
- "S11 = −9.6 dB > −10 dB → 通带匹配不足 → 将 ripple_db 从 0.5 降至 0.25"
- "阻带衰减 25 dB < 目标 40 dB → 将阶数从 3 增至 5"

---

## IV. Experiments / 实验

### IV-A. Setup / 实验设置

#### EN

**Data:** 200 random test samples (175 full + 8 followup-Q + 16 followup-R) for comprehensive evaluation; 100 samples for ablation comparison.

**Metrics:** (1) JSON parse rate; (2) filter_type accuracy; (3) filter_band accuracy; (4) **order accuracy** $\text{Acc}_N = \mathbb{1}[N_{\text{pred}} = N_{\text{gt}}]$ (core metric); (5) numerical relative error $\epsilon_k = |v_{\text{pred}} - v_{\text{gt}}| / v_{\text{gt}}$.

**Baseline:** Original Qwen3-8B with identical system prompt and template.

#### 中文

**数据**：200 样本全面评估（175 full + 8 followup-Q + 16 followup-R），100 样本消融对比。**指标**：JSON 解析率、类型/频段准确率、**阶数准确率**（核心）、数值相对误差。**基线**：未微调 Qwen3-8B（相同提示词和模板）。

---

### IV-B. Main Results / 主要结果

#### EN

**Table III: Qwen3-8B-RF Evaluation (200 samples) / 模型评估（200 样本）**

| Metric | Result |
|:---|:---:|
| JSON Parse Rate | **100.0%** |
| Filter Type Accuracy | **100.0%** |
| Filter Band Accuracy | **100.0%** |
| **Order Accuracy (Overall)** | **83.4%** |
| — LPF | 98.6% |
| — HPF | 98.2% |
| — BPF | 47.1% |
| Numerical Median Error | ≈ 0.000% |

**Key findings:** (1) 100% JSON compliance — all outputs directly parseable by EDA tools. (2) LPF/HPF order accuracy near-perfect (98.6%, 98.2%). (3) BPF accuracy (47.1%) reflects higher-dimensional complexity. (4) Zero median numerical error — the model learns exact parameter reproduction.

#### 中文

见表 III。核心发现：(1) JSON 遵从率 100%；(2) LPF/HPF 阶数准确率接近完美（98.6%, 98.2%）；(3) BPF 准确率 47.1% 反映高维复杂性；(4) 数值中位误差为零——模型学会精确参数还原。

---

### IV-C. Ablation Study / 消融实验

#### EN

We conduct a three-way ablation comparing: (1) the unmodified baseline Qwen3-8B, (2) QLoRA fine-tuned on an early-stage dataset without data augmentation (train_q4_24g_safe, ~11k samples), and (3) our full pipeline with cleaned data and augmentation strategies (train_cleaned_v2, ~25k samples including sensitivity analysis, reflection, and curriculum-ordered data).

**Table IV: Three-Way Ablation / 三方消融实验**

| Metric | Qwen3-8B (No FT) | QLoRA-Early (~11k) | **QLoRA-Full (Ours, ~25k)** |
|:---|:---:|:---:|:---:|
| JSON Parse | 100.0% | 100.0% | **100.0%** |
| Filter Type | 100.0% | 95.4% ↓ | **100.0%** |
| Filter Band | 100.0% | 100.0% | **100.0%** |
| **Order Accuracy** | **19.5%** | **12.6%** ↓ | **83.4%** |
| Followup (ask) | N/A | 0.0% | 25.0% |

**Table V: Order Accuracy by Band / 分频段阶数准确率**

| Band | Qwen3-8B (No FT) | QLoRA-Early | **QLoRA-Full (Ours)** |
|:---:|:---:|:---:|:---:|
| LPF | 26.3% | 7.9% ↓ | **98.6%** |
| HPF | 23.3% | 10.0% ↓ | **98.2%** |
| BPF | 0.0% | 26.3% | **47.1%** |

**Critical finding: naive fine-tuning can *hurt* performance.** The QLoRA-Early model, trained on ~11k samples from an uncleaned earlier dataset without augmentation, achieves only 12.6% order accuracy — **worse than the unfine-tuned baseline** (19.5%). This reveals two important insights:

1. **Data quality > model training**: Fine-tuning on a small, un-curated dataset introduces noise that degrades the base model's general reasoning ability without compensating with sufficient domain knowledge. The filter_type accuracy drops from 100% to 95.4%, indicating the noisy data even corrupts basic classification.

2. **Data augmentation is essential**: Our full pipeline (C1) includes three augmentation strategies — sensitivity analysis samples (2,000), reflection/correction pairs (5,553), and reverse-prediction tasks (2,000) — that collectively 2.3× the training data (11k → 25k) and improve diversity. The order accuracy jumps from 12.6% to 83.4%, a **70.8 percentage point improvement over naive fine-tuning**.

The three-tier difficulty gradient (LPF 98.6% > HPF 98.2% >> BPF 47.1%) correlates with parameter-space dimensionality: LPF/HPF depend on 2 variables ($k_s, L_A$), while BPF depends on 5 ($f_0, \Delta f, f_{s,\text{lower}}, f_{s,\text{upper}}, L_A$).

#### 中文

三方消融实验对比见表 IV–V：(1) 未微调 Qwen3-8B，(2) 早期小数据集 (~11k) 的 QLoRA 微调，(3) 完整数据增强 (~25k) 的 QLoRA 微调。

**关键发现：朴素微调反而损害性能。** QLoRA-Early 模型仅达 12.6% 阶数准确率——**低于未微调基线** (19.5%)。这揭示两点：(1) **数据质量 > 模型训练**：小规模未清洗数据引入噪声，破坏了基础模型的通用推理能力；(2) **数据增强不可或缺**：完整管线包含灵敏度分析 (2,000)、反思纠错 (5,553)、逆向预测 (2,000) 三种增强策略，将数据量提升 2.3 倍 (11k → 25k)，阶数准确率从 12.6% 跃升至 83.4%，**相比朴素微调提升 70.8 个百分点**。

---

### IV-D. S-Parameter Validation / S 参数验证

#### EN

To demonstrate *why* order accuracy matters for actual circuit performance, we computed S-parameter frequency responses via ABCD matrix chain multiplication for correctly vs. incorrectly predicted filter orders.

**Fig. 2** (`sparam_lpf_comparison.pdf`): Chebyshev LPF, $f_c$=1 GHz, $f_s$=2 GHz, $L_r$=0.1 dB. The correct order N=5 achieves ≥42 dB stopband attenuation, while N=3 (typical baseline prediction) yields only ~18 dB — **failing the spec by >20 dB**.

**Fig. 3** (`sparam_bpf_comparison.pdf`): Chebyshev BPF, $f_0$=1.5 GHz, BW=200 MHz. N=7 meets requirements; N=4 shows significant performance gaps.

**Fig. 4** (`order_impact_sparam.pdf`): Parametric sweep of $|S_{21}|$ for N=2 to 9. The minimum order satisfying $L_A$≥40 dB is N=5 — precisely the value our model learns to predict.

#### 中文

图 2–4 展示阶数对实际电路性能的影响。以 LPF ($f_c$=1 GHz, $f_s$=2 GHz) 为例：正确阶数 N=5 满足 ≥42 dB 阻带衰减；基线典型预测 N=3 仅达 ~18 dB，差距 >20 dB。图 4 阶数扫描表明 N≥5 是满足指标的最低阶——正是微调模型学习到的映射。

---

### IV-E. Closed-Loop Agent Validation / 闭环 Agent 验证

#### EN

We validated the agent on a representative case: Chebyshev LPF, $f_c$=1 GHz, $f_s$=2 GHz, ripple=0.5 dB, $R_0$=50 Ω. The agent converged in **2 iterations**:

| Iter | ripple_db | S11 (dB) | Passband Ripple (dB) | Stopband Atten (dB) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.50 | −9.6 | 0.50 | −42.0 | **FAIL** (S11) |
| 2 | 0.25 | −12.4 | 0.26 | −38.9 | **PASS** |

The agent diagnosed: *"S11 = −9.6 dB exceeds −10 dB threshold → passband matching insufficient → autonomously reduce ripple_db from 0.5 to 0.25."* This demonstrates physics-informed reflective reasoning — the model understands the causal relationship between ripple and return loss.

#### 中文

Agent 在 Chebyshev LPF ($f_c$=1 GHz, ripple=0.5 dB) 上 **2 轮**收敛。首轮 S11=−9.6 dB 不满足 <−10 dB 要求；Agent 自主诊断"通带匹配不足"，将 ripple 从 0.5 降至 0.25 dB，第 2 轮 S11=−12.4 dB 通过。展示了基于物理语义的因果推理。

---

### IV-F. Comparison with Related Work / 与相关工作对比

#### EN

**Table VI: Comparison with Existing Methods / 与已有方法对比**

| Method | Base Model | Fine-tuned? | Bands | Order Acc. | JSON | Local | Closed-Loop | Cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| WiseEDA [1] | GPT-4 | ✗ | LPF | N/R† | N/R | ✗ | ✓ (PSO) | API cost |
| ChipChat [2] | ChatGPT | ✗ | N/A‡ | N/A | N/A | ✗ | ✗ | API cost |
| ChipGPT [3] | GPT-3.5 | ✗ | N/A‡ | N/A | N/A | ✗ | ✗ | API cost |
| Qwen3-8B | Qwen3-8B | ✗ | LPF/HPF/BPF | 19.5% | 100% | ✓ | ✗ | — |
| **Ours** | **Qwen3-8B** | **✓ QLoRA** | **LPF/HPF/BPF** | **83.4%** | **100%** | **✓** | **✓ (Reflect)** | **9.4h 1×4090** |

†N/R = Not reported. WiseEDA focuses on PSO optimization integration, not order prediction accuracy.
‡N/A = Not applicable. ChipChat/ChipGPT target digital circuits (Verilog/VHDL), not RF filter synthesis.

**Key differentiators of RF-FilterLLM / 关键差异化优势：**

1. **First fine-tuned RF-specific LLM** — all others use prompt engineering only / 首个 RF 领域微调 LLM
2. **Multi-band in one model** — LPF + HPF + BPF / 单模型三频段
3. **Quantitative order metric** — first to benchmark $\text{Acc}_N$ / 首次量化阶数指标
4. **Fully local deployment** — no API, IP-safe, 15.6 GB model / 完全本地，保护 IP
5. **Physics-informed reflection** — causal reasoning, not blind search / 因果推理，非盲目搜索

#### 中文

见表 VI。RF-FilterLLM 的五个关键差异化优势：首个 RF 微调 LLM、单模型三频段、首次量化阶数指标、完全本地部署（15.6 GB）、基于物理语义的因果推理。

---

## V. Discussion / 讨论

### EN

**Why order accuracy is the critical metric.** Format compliance, type/band classification are solved by modern LLMs without fine-tuning (all 100%). Order prediction requires internalizing $N = \lceil f(f_c, f_s, L_r, L_A) \rceil$ — a nonlinear ceiling function. This is precisely where domain-specific SFT provides value, and why we propose $\text{Acc}_N$ as the primary evaluation metric for LLM-based filter design.

**BPF: an open challenge.** BPF accuracy (47.1%) lags LPF/HPF due to: (1) 5D vs. 2D parameter space; (2) narrowband approximation nonlinearity; (3) fewer training samples (30.1%). Mitigation strategies include BPF oversampling, curriculum learning [7], reverse-prediction auxiliary tasks, and chain-of-thought reasoning for multi-step frequency transformations.

**Efficiency and accessibility.** QLoRA reduces trainable parameters to 0.047%, enabling training on a single consumer GPU (RTX 4090) in under 10 hours. The merged model (15.6 GB BF16) runs inference on the same hardware. This democratizes domain-specific EDA assistance for small teams and academic labs.

**Limitations.** (1) Only Chebyshev/Butterworth supported; elliptic and Bessel filters planned. (2) Follow-up question samples (3.9%) are limited. (3) Closed-loop agent tested on limited cases; larger-scale systematic evaluation needed. (4) Only Qwen3-8B validated; cross-model transfer remains unexplored.

### 中文

**为何阶数准确率是核心指标** — 格式/类型/频段均已被通用 LLM 解决（100%）。阶数预测需内化非线性取整函数 $N = \lceil f(f_c, f_s, L_r, L_A) \rceil$，这正是领域 SFT 的价值所在。

**BPF 仍是开放挑战** — 准确率 47.1%，因 5D 参数空间、窄带近似非线性、训练样本不足 (30.1%)。可通过过采样、课程学习 [7]、反向预测辅助任务和思维链推理改善。

**效率与可及性** — QLoRA 将可训练参数压缩至 0.047%，单张 4090 训练 <10 小时，合并模型 15.6 GB 即可推理。使小型团队和学术实验室也能构建领域 EDA 助手。

**局限** — (1) 仅支持 Chebyshev/Butterworth；(2) 追问样本有限 (3.9%)；(3) 闭环 Agent 测试案例有限；(4) 仅验证 Qwen3-8B，跨模型迁移待探索。

---

## VI. Conclusion / 结论

### EN

We presented RF-FilterLLM, a framework for automated RF filter design through efficient LLM fine-tuning. With the 25,168-sample RF-Filter-SFT dataset and QLoRA fine-tuning (0.047% parameters, 9.4h on 1× RTX 4090), we improved filter-order prediction from 19.5% to 83.4%, with 100% JSON compliance and near-zero numerical error. The closed-loop reflective agent demonstrated physics-informed iterative optimization with ADS simulation, converging in 2 iterations. Our results establish domain-specific instruction tuning — not prompt engineering alone — as essential for unlocking LLM potential in RF EDA.

### 中文

本文提出 RF-FilterLLM，基于 25,168 样本的 RF-Filter-SFT 数据集和 QLoRA 微调 (0.047% 参数, 9.4h, 1×4090)，将阶数预测准确率从 19.5% 提升至 83.4%，JSON 遵从率 100%，数值误差趋近于零。闭环反思 Agent 与 ADS 集成，2 轮迭代收敛。实验证明：领域指令微调——而非仅提示工程——是释放 LLM 在 RF EDA 领域潜力的关键。

---

## References

[1] Y. Zhang *et al.*, "WiseEDA: An LLM-assisted automated design framework for RF circuits," *IEEE Trans. Microw. Theory Techn.*, 2024.

[2] J. Blocklove *et al.*, "Chip-Chat: Challenges and opportunities in conversational hardware design," in *Proc. ACM/IEEE DAC*, 2023.

[3] K. Chang *et al.*, "ChipGPT: How far are we from natural language hardware design," *arXiv:2305.14019*, 2023.

[4] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.

[5] T. Dettmers *et al.*, "QLoRA: Efficient finetuning of quantized language models," in *Proc. NeurIPS*, 2023.

[6] G. L. Matthaei, L. Young, and E. M. T. Jones, *Microwave Filters, Impedance-Matching Networks, and Coupling Structures*. Norwood, MA: Artech House, 1980.

[7] Y. Bengio *et al.*, "Curriculum learning," in *Proc. ICML*, 2009.

[8] N. Shinn *et al.*, "Reflexion: Language agents with verbal reinforcement learning," in *Proc. NeurIPS*, 2023.

[9] Y. Zheng *et al.*, "LlamaFactory: Unified efficient fine-tuning of 100+ language models," in *Proc. ACL System Demonstrations*, 2024.

---

## System Architecture Diagram / 系统架构图

> **Fig. 1** 需手绘（推荐 draw.io / PowerPoint / Visio）。以下为架构说明和 ASCII 原型：

```
┌────────────────────────────────────────────────────────────┐
│              RF-FilterLLM System Architecture               │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Module A     │  │  Module B     │  │  Module C         │  │
│  │  Data Engine  │  │  QLoRA Train  │  │  Closed-Loop Agent│  │
│  │              │  │              │  │                  │  │
│  │  Param Space  │  │  Qwen3-8B    │  │  User Request    │  │
│  │  Sampling     │  │  ↓ NF4       │  │  ↓               │  │
│  │  ↓            │  │  ↓ LoRA r=8  │  │  LLM Inference   │  │
│  │  Order Calc   │  │  → Training  │  │  (JSON output)   │  │
│  │  N=⌈Eq.1⌉    │  │    3146 steps│  │  ↓               │  │
│  │  ↓            │  │    loss=0.28 │  │  ADS Simulation  │  │
│  │  Prototype    │  │              │  │  (hpeesofsim)    │  │
│  │  Synthesis gk │  │  Output:     │  │  ↓               │  │
│  │  ↓            │  │  Qwen3-8B-RF │  │  S-param Eval    │  │
│  │  ShareGPT     │  │  (15.6 GB)   │  │  PASS → Return   │  │
│  │  Formatting   │  │              │  │  FAIL ↓           │  │
│  │  ↓            │  │              │  │  Reflect & Fix   │  │
│  │  RF-Filter-SFT│══▶              ══▶│  (≤5 iters)      │  │
│  │  25,168 train │  │              │  │  ↻ Loop back     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                            │
│  Data Flow:  A ══▶ B ══▶ C                                │
│  Innovation: C1(Dataset) C2(QLoRA) C3(Multi-band) C4(Agent)│
└────────────────────────────────────────────────────────────┘
```

**绘制要点 / Drawing Guide:**
1. 三个模块用不同颜色底色（蓝 A / 绿 B / 橙 C）
2. Module A → B 粗箭头标注"25,168 samples"
3. Module B → C 粗箭头标注"Qwen3-8B-RF (15.6 GB)"
4. Module C 中闭环用带箭头循环表示（FAIL → Reflect → Re-generate）
5. 标注四个创新点 C1–C4 的对应位置

---

## Figure & Table Index / 图表索引

| Figure | Content | File |
|:---:|:---|:---|
| 1 | System architecture (manual) | *Draw.io / PPT* |
| 2 | LPF S-param: correct vs. wrong order | `sparam_lpf_comparison.pdf` |
| 3 | BPF S-param: correct vs. wrong order | `sparam_bpf_comparison.pdf` |
| 4 | Order parametric sweep $\|S_{21}\|$ | `order_impact_sparam.pdf` |
| 5 | Training loss curve | `training_loss_curve.pdf` |
| 6 | Baseline comparison bar chart | `baseline_comparison.pdf` |
| 7 | Order accuracy by band | `order_accuracy_by_band.pdf` |
| 8 | Dataset distribution | `dataset_distribution.pdf` |
| 9 | Evaluation radar chart | `evaluation_radar.pdf` |
| 10 | Numerical error box plot | `numerical_errors.pdf` |

| Table | Content |
|:---:|:---|
| I | RF-Filter-SFT dataset statistics |
| II | QLoRA training configuration |
| III | Overall evaluation (200 samples) |
| IV | Ablation: baseline vs. fine-tuned (100 samples) |
| V | Order accuracy by filter band |
| VI | Comparison with related work |
