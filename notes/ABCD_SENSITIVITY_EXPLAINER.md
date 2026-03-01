# ABCD 矩阵扰动灵敏度分析：原理详解

> 对应论文公式 (2) 及 `sensitivity_analysis.py` 中的 `SensitivityAnalyzer` 类。

---

## 1. 什么是 ABCD 矩阵？

### 1.1 二端口网络的统一描述

任何一个"两端进、两端出"的电路（二端口网络）都可以用一个 2×2 矩阵来描述其端口电压/电流关系：

$$\begin{bmatrix} V_1 \\ I_1 \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} V_2 \\ I_2 \end{bmatrix}$$

其中 $V_1, I_1$ 是输入端电压/电流，$V_2, I_2$ 是输出端电压/电流。

ABCD 矩阵（也叫"传输矩阵"或"链矩阵"）的最大优点是：**级联电路 = 矩阵连乘**。

$$M_{\text{total}} = M_1 \cdot M_2 \cdot M_3 \cdots M_N$$

### 1.2 梯形滤波器的基本元件

切比雪夫梯形网络由两种基本元件交替构成：

| 元件 | 位置 | ABCD 矩阵 | 阻抗/导纳 |
|------|------|-----------|-----------|
| 串联电感 $L_k$ | 奇数位（$k=1,3,5,\ldots$）| $\begin{bmatrix}1 & Z\\0 & 1\end{bmatrix}$ | $Z = j\omega L_k$ |
| 并联电容 $C_k$ | 偶数位（$k=2,4,6,\ldots$）| $\begin{bmatrix}1 & 0\\Y & 1\end{bmatrix}$ | $Y = j\omega C_k$ |

代码中使用**归一化原型**（$\omega_c = 1$，$R_0 = 1$），直接用 $g_k$ 代替实际 $L_k/C_k$，两者关系为：

$$L_k = \frac{R_0 \cdot g_k}{\omega_c}, \quad C_k = \frac{g_k}{R_0 \cdot \omega_c}$$

---

## 2. 从 ABCD 矩阵到 S21

### 2.1 链矩阵级联

对于一个 $N$ 阶梯形网络，从端口 1 到端口 N 依次相乘得到总传输矩阵 $[A, B; C, D]$：

```
源 (R_S=1) ─── [g1: 串联] ─── [g2: 并联] ─── ··· ─── [gN] ─── 负载 (R_L)
```

初始为单位矩阵，然后：
- 遇到串联元件（$k$ 为奇数）：右乘 $\begin{bmatrix}1 & j\omega g_k\\0 & 1\end{bmatrix}$
- 遇到并联元件（$k$ 为偶数）：右乘 $\begin{bmatrix}1 & 0\\j\omega g_k & 1\end{bmatrix}$

```python
# sensitivity_analysis.py 第 63-79 行
for k, g in enumerate(gk):
    if k % 2 == 0:          # 串联电感
        Z = s * g           # s = jω（归一化）
        A_new = A + C * Z
        B_new = B + D * Z
        A, B = A_new, B_new
    else:                   # 并联电容
        Y = s * g
        C_new = C + A * Y
        D_new = D + B * Y
        C, D = C_new, D_new
```

### 2.2 由 ABCD 计算 S21

得到总传输矩阵后，传输系数 $S_{21}$ 为：

$$S_{21} = \frac{2\sqrt{R_S R_L}}{A \cdot R_L + B + C \cdot R_S R_L + D \cdot R_S}$$

在归一化条件下（$R_S = 1$）：

$$S_{21} = \frac{2\sqrt{R_L}}{A \cdot R_L + B + C \cdot R_L + D}$$

功率传输比（即什么比例的输入功率被传送到负载）：

$$|S_{21}|^2 = \left|\frac{2\sqrt{R_L}}{A R_L + B + C R_L + D}\right|^2$$

阻带衰减（dB）：

$$L_A = -10\log_{10}|S_{21}(\omega_s)|^2$$

通带回波损耗（S11）由功率守恒得：

$$|S_{11}|^2 = 1 - |S_{21}|^2$$

---

## 3. 灵敏度的定义

### 3.1 归一化灵敏度

论文公式 (2)：

$$S_{x_i}^{m} = \frac{m\bigl(x_i(1+\delta)\bigr) - m(x_i)}{m(x_i) \cdot \delta}, \quad \delta = 1\%$$

其中：
- $x_i$：第 $i$ 个元件的值（$L_k$ 或 $C_k$，通过 $g_k$ 参数化）
- $m(x_i)$：以 $x_i$ 为参数时的某个性能指标（阻带衰减/S11/通带纹波）
- $\delta = 0.01$（1% 微扰）

直觉解读：**如果把 $x_i$ 变大 1%，指标 $m$ 会变化百分之几？**

例如 $S_{L_1}^{L_A} = 5$ 意味着：$L_1$ 增大 1% → 阻带衰减增大 5%（极高灵敏度）。

### 3.2 综合影响分数

对每个元件，将其对所有指标的灵敏度绝对值加总：

$$\text{abs\_impact}(x_i) = \left|S_{x_i}^{L_A}\right| + \left|S_{x_i}^{S_{11}}\right| + \left|S_{x_i}^{L_r}\right|$$

按 `abs_impact` 从大到小排名，分为三档：
- **High**（前 1/3）：优先调整
- **Medium**（中 1/3）：次优先
- **Low**（后 1/3）：基本不动

---

## 4. 完整计算流程

```
给定设计 (N, ripple_db, fc, fs, R0)
        │
        ▼
计算原始 gk → 基线指标 (baseline)
  - stopband_atten_dB = -10 log₁₀|S21(ωs)|²
  - S11_max_dB = 10 log₁₀(1 - min|S21|²)
  - passband_ripple_dB = -10 log₁₀(min|S21|²) in [0,1]
        │
        ├── 对每个 L1, L2, ... 施加 +1% 扰动
        │   └── 仅修改对应 gk，重新计算 ABCD → 新指标 → 算灵敏度
        │
        └── 对每个 C1, C2, ... 施加 +1% 扰动
            └── 仅修改对应 gk，重新计算 ABCD → 新指标 → 算灵敏度
        │
        ▼
按 abs_impact 排名 → 生成优先级报告
        │
        ▼
输出给 LLM：
  "L3 (rank=1, high priority): sensitivity=12.4 — 优先调整"
  "C2 (rank=2, high priority): sensitivity=9.7"
  ...
```

---

## 5. 为什么用 ABCD 矩阵而不是直接调用 ADS？

| 对比项 | ABCD 矩阵（本文方法）| ADS 仿真 |
|--------|---------------------|----------|
| 速度 | 微秒级（纯 NumPy） | 秒~分钟级 |
| 调用次数 | 每个元件 1 次（N 次总计）| 同样需要 N 次 |
| 精度 | 理想电路精确 | 含寄生效应，更真实 |
| 用途 | 训练数据生成（海量）| 闭环验证（少量）|

在**训练阶段**，需要为数万条样本生成灵敏度标注，ABCD 方法使这在单机上可行。在**推理阶段**（闭环），最终设计验证仍使用 ADS。

---

## 6. 具体数值举例

以 5 阶低通滤波器为例（$f_c=1\,\text{GHz}$，ripple=0.5 dB，$L_A=40\,\text{dB}$）：

原始 $g_k = [0.7563,\ 1.3049,\ 1.5773,\ 1.3049,\ 0.7563]$

**对 $g_3$（L3，中间电感）施加 +1% 扰动**：

$$g_3' = 1.5773 \times 1.01 = 1.5931$$

重新算 ABCD → $|S_{21}(\omega_s)|^2$ 变化 → 阻带衰减从 40.0 dB → 41.3 dB

$$S_{L_3}^{L_A} = \frac{41.3 - 40.0}{40.0 \times 0.01} = \frac{1.3}{0.4} = 3.25$$

**对比 $g_1$（L1，端口电感）施加 +1% 扰动**：

阻带衰减几乎不变（端口元件主要影响匹配，不影响阻带截止），$S_{L_1}^{L_A} \approx 0.2$

→ 结论：**中间元件（L3/C3）灵敏度最高**，闭环修正时 LLM 应优先调整阶数或中间参数。

---

## 7. 与论文其他模块的关系

```
灵敏度报告
    │
    ├──▶ 训练数据 (sensitivity_train.jsonl)
    │      "哪些元件最关键" → 让 LLM 学会"射频直觉"
    │
    └──▶ 闭环提示词 (llm_ads_loop_iterative.py)
           "rank=1: L3 sensitivity=12.4" 注入 reflection prompt
           → LLM 知道重点在哪里，调整更精准，收敛更快
```

---

*本文档对应代码：`sensitivity_analysis.py`，论文公式 (2)。*
