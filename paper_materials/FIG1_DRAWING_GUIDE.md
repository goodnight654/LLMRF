# Fig. 1 系统架构图 — 手绘指南

## 推荐工具
- **draw.io** (https://app.diagrams.net) — 免费、导出 PDF/PNG/SVG
- PowerPoint / Visio 亦可

## 整体布局

**画布尺寸**: 宽 24cm × 高 14cm（IEEE 双栏论文 column width ≈ 8.8cm，但系统架构图通常跨双栏使用 ~18cm）

**三列横向排列**:

```
┌────────────┐  ┌────────────┐  ┌──────────────────┐
│  Module A   │  │  Module B   │  │  Module C          │
│  Data Engine│→ │  QLoRA Train│→ │  Closed-Loop Agent │
│  (蓝色底)   │  │  (绿色底)   │  │  (橙色底)           │
└────────────┘  └────────────┘  └──────────────────┘
     ~7cm            ~6cm             ~9cm
```

---

## 配色方案

| 模块 | 底色 (HEX) | 边框 (HEX) | 标题文字 |
|:---:|:---:|:---:|:---:|
| Module A | `#E3F2FD` 浅蓝 | `#1565C0` 深蓝 | 白底蓝字 |
| Module B | `#E8F5E9` 浅绿 | `#2E7D32` 深绿 | 白底绿字 |
| Module C | `#FFF3E0` 浅橙 | `#E65100` 深橙 | 白底橙字 |
| 数据库圆柱 | `#BBDEFB` (A用), `#C8E6C9` (B用) | 同边框色 | — |
| PASS 圆角矩形 | `#A5D6A7` 浅绿 | `#2E7D32` | — |
| FAIL 圆角矩形 | `#FFCDD2` 浅红 | `#C62828` | — |
| 粗箭头 | `#424242` 深灰 | — | 白底标注 |

---

## Module A — Data Engine 数据引擎 (蓝色区域)

### 框内元素（从上往下排列）

1. **标题栏** (矩形, 深蓝底白字)
   - `Module A: Data Engine 数据引擎`

2. **圆角矩形 ①** — 参数空间采样
   - 文字: `Parameter Space Sampling`
   - 副文字(小号): `fc∈[100M, 3G]  ks∈[1.2, 3.0]`
   - 副文字(小号): `Lr∈{0.01,...,1.0}  LA∈[20, 60]`

3. **↓ 箭头**

4. **圆角矩形 ②** — 阶数计算
   - 文字: `Order Computation`
   - 公式(斜体): `N = ⌈cosh⁻¹(···)/cosh⁻¹(ks)⌉`

5. **↓ 箭头**

6. **圆角矩形 ③** — 原型综合
   - 文字: `Prototype Synthesis`
   - 副文字: `gk values → L, C elements`

7. **↓ 箭头**

8. **圆角矩形 ④** — 对话格式封装
   - 文字: `ShareGPT Formatting`
   - 三行小字 (用不同灰度区分):
     - `Full: 86.8%`
     - `Followup-Q: 3.9%  (C1: 动态意图补全)`
     - `Followup-R: 9.3%`

9. **↓ 箭头**

10. **圆柱形 (数据库图标)** — 数据集
    - 文字: `RF-Filter-SFT`
    - 副文字: `25,168 train / 1,415 val / 1,372 test`
    - 颜色: `#BBDEFB` 底

---

## Module B — QLoRA Training (绿色区域)

### 框内元素

1. **标题栏** (矩形, 深绿底白字)
   - `Module B: QLoRA Training`

2. **圆角矩形 ①** — 基础模型
   - 文字: `Qwen3-8B`
   - 副文字: `8.19B params, 36 layers`

3. **↓ 箭头**

4. **圆角矩形 ②** — 量化
   - 文字: `NF4 Quantization (4-bit)`

5. **↓ 箭头**

6. **圆角矩形 ③** — LoRA 注入 ★关键标注
   - 文字: `LoRA Injection`
   - 副文字: `r=8, α=16`
   - 副文字: `q_proj, v_proj`
   - 右侧小标: `3.83M (0.047%)` ← 用红色/粗体强调

7. **↓ 箭头**

8. **圆角矩形 ④** — 训练
   - 文字: `Training (LLaMA-Factory)`
   - 副文字: `1× RTX 4090, 9.4h`
   - 副文字: `3,146 steps, loss=0.28`

9. **↓ 箭头**

10. **圆柱形** — 输出模型
    - 文字: `Qwen3-8B-RF`
    - 副文字: `Merged 15.6 GB`
    - 颜色: `#C8E6C9` 底

---

## Module C — Closed-Loop Agent (橙色区域)

### 框内元素 (关键：有循环箭头！)

1. **标题栏** (矩形, 深橙底白字)
   - `Module C: Closed-Loop Agent 闭环Agent`

2. **圆角矩形 ①** — 用户需求
   - 文字: `User Specification`
   - 副文字: `Natural language input`
   - 图标: 可选用"人形"图标

3. **↓ 箭头**

4. **圆角矩形 ②** — LLM 推理 ★ 核心
   - 文字: `LLM Inference`
   - 副文字: `Qwen3-8B-RF → JSON`
   - **注意**: 从 Module B 的输出有一条粗箭头指向此处
   - **注意**: 从下方"反思修正"有一条**红色虚线回路箭头**指回此处

5. **↓ 箭头**

6. **圆角矩形 ③** — ADS 仿真
   - 文字: `ADS Simulation`
   - 副文字: `hpeesofsim API`
   - 副文字: `→ S11, S21 data`

7. **↓ 箭头**

8. **菱形** — 评估判断 ★ 用菱形（draw.io: Decision shape）
   - 文字: `Evaluate`
   - 条件: `S11 < -10 dB?`
   - 条件: `Ripple ≤ Lr?`

9. **右侧出口 → PASS**
   - 圆角矩形 (绿底): `✓ PASS → Return Design`

10. **下方出口 → FAIL**
    - 圆角矩形 (红底): `✗ FAIL`

11. **↓ 从 FAIL**

12. **圆角矩形 ⑤** — 反思修正 ★ 关键创新
    - 文字: `Reflect & Fix 反思修正`
    - 副文字: `Physics-aware reasoning`
    - 副文字: `≤ 5 iterations`
    - 标注: `Innovation C4` (用星号或徽章)

13. **↑ 回路箭头** (红色虚线, 从 ⑤ 回到 ②)
    - 标注: `Corrected params`
    - 颜色: `#C62828` 红色

---

## 模块间连接箭头

### A → B (数据流)
- **粗箭头** 从 Module A 的"RF-Filter-SFT 圆柱"→ Module B 的"Qwen3-8B 矩形"
- 箭头上方文字: `Training Data`
- 箭头下方数字: `25,168 samples`
- 颜色: `#424242` 深灰, 线宽 3px

### B → C (模型部署)
- **粗箭头** 从 Module B 的"Qwen3-8B-RF 圆柱"→ Module C 的"LLM Inference 矩形"
- 箭头上方文字: `Model Deploy`
- 箭头下方数字: `15.6 GB BF16`
- 颜色: `#424242` 深灰, 线宽 3px

---

## 标注四个创新点

在图底部或各模块角落用**徽章/标签**标注创新点:

| 创新 | 位置 | 文字 | 颜色 |
|:---:|:---|:---|:---:|
| C1 | Module A 的 ShareGPT Formatting 旁 | `C1: Dynamic Intent Completion` | 蓝色 |
| C2 | Module B 的 LoRA Injection 旁 | `C2: 0.047% Parameter Efficient` | 绿色 |
| C3 | Module A 底部或 B-C 之间 | `C3: Multi-Band LPF/HPF/BPF` | 紫色 |
| C4 | Module C 的 Reflect & Fix 旁 | `C4: Physics-Informed Reflection` | 橙色 |

---

## draw.io 具体操作步骤

1. 打开 https://app.diagrams.net/
2. 选择「Blank Diagram」→ 存储为 `rf_filterlm_architecture.drawio`
3. **画大框**: 用「Rectangle」画 3 个大矩形，设置底色和 2px 边框
4. **画内部元素**: 用「Rounded Rectangle」画流程步骤
5. **画数据库**: 用「Cylinder」图标画数据集和模型
6. **画菱形**: 用「Decision」(搜索 "diamond") 画评估节点
7. **画箭头**: 用 connector 连接，A→B 和 B→C 用粗箭头 (Style → Line → 3px)
8. **画回路**: C 中 Reflect → LLM Inference 用红色虚线箭头
9. **导出**: File → Export as → PDF (勾选 "Crop")

---

## 导出设置

- **PDF**: File → Export as → PDF → Crop → Page fit
- **PNG**: File → Export as → PNG → DPI=300, Background=White
- 保存到: `paper_materials/latex/figs/system_architecture.pdf`

---

## 参考: Mermaid 版本已生成

已在 VS Code 中用 Mermaid 渲染了一个简化版，可作为布局参考。
Mermaid 版本特点: 三模块分色 (蓝/绿/橙)、数据流箭头、闭环回路。
手绘版可在此基础上添加更多细节和标注。
