r"""
性能预测预训练数据生成模块 (Performance Prediction Pre-training)
创新点实现：反向预测任务
核心思想：
  正向任务: 给定性能指标 → 输出电路参数 (标准 SFT)
  反向任务: 给定电路参数 → 预测性能指标 (本模块)

  通过双向学习，模型能建立"参数↔性能"的双向映射，
  类似人类工程师的"直觉"——看到一组元件值就能大致估计性能。
运行:
  python generate_reverse_prediction_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from adsapi.filter_designer import (
    ChebyshevFilterDesigner,
    ButterworthFilterDesigner,
    get_filter_designer,
)


SYSTEM_PROMPT_PREDICT = (
    "你是射频滤波器性能预测专家。"
    "给定滤波器的电路参数（电感、电容值、阶数、滤波器类型），"
    "你需要预测其关键性能指标，包括阻带衰减、通带纹波、S11等。"
    "输出必须是可解析JSON。"
)

SYSTEM_PROMPT_COMPARE = (
    "你是射频滤波器设计比较专家。"
    "给定两组滤波器参数及其对应的仿真结果，"
    "你需要分析并比较两者的优劣，给出结论。"
    "输出必须是可解析JSON。"
)


def _compute_performance(design: Dict, spec: Dict) -> Dict[str, Any]:
    """基于公式计算性能指标"""
    N = design['N']
    ripple_db = spec.get('ripple_db', design['params']['ripple_db'])
    fc = spec.get('fc', design['params'].get('fc', 1e9))
    fs = spec.get('fs', design['params'].get('fs', 2e9))
    R0 = spec.get('R0', design['params']['R0'])
    ep = 10 ** (ripple_db / 10) - 1
    wc = 2 * np.pi * fc
    ws = 2 * np.pi * fs

    # 阻带衰减
    try:
        atten = 10 * np.log10(1 + ep * np.cosh(N * np.arccosh(ws / wc)) ** 2)
    except (ValueError, FloatingPointError):
        atten = 0.0

    # S11
    if ripple_db > 0:
        s11_max = 10 * np.log10(max(1e-20, 1 - 10 ** (-ripple_db / 10)))
    else:
        s11_max = -40.0

    # 3dB 截止频率偏差 (理想为 0)
    fc_deviation_pct = 0.0  # 公式设计下精确匹配

    # 群延时估计 (简化)
    tau_approx_ns = N / (2 * np.pi * fc) * 1e9

    return {
        "stopband_attenuation_dB": round(float(atten), 2),
        "passband_ripple_dB": round(float(ripple_db), 3),
        "S11_max_dB": round(float(s11_max), 2),
        "order": N,
        "fc_deviation_pct": round(float(fc_deviation_pct), 2),
        "group_delay_ns": round(float(tau_approx_ns), 3),
    }


def generate_forward_prediction_sample(
    rng: random.Random,
    filter_type: str = "chebyshev",
) -> Optional[Dict]:
    """
    生成正向预测样本: 给定元件值 → 预测性能

    user: "以下滤波器的元件值为 L1=xx nH, C1=xx pF, ...，请预测性能"
    assistant: {"stopband_attenuation_dB": xx, "S11_max_dB": xx, ...}
    """
    ripple = rng.choice([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    fc = rng.choice([400e6, 600e6, 800e6, 1e9, 1.2e9, 1.5e9, 2e9, 2.5e9])
    fs_ratio = rng.uniform(1.3, 3.0)
    fs = fc * fs_ratio
    La = rng.uniform(25, 60)
    R0 = rng.choice([25, 50, 75, 100])
    order = rng.randint(3, 9)

    designer = get_filter_designer(filter_type)
    try:
        design = designer.design_lpf_by_attenuation(
            ripple_db=ripple, fc=fc, fs=fs, R0=R0, La=La, order=order
        )
    except Exception:
        return None

    spec = {'ripple_db': ripple, 'fc': fc, 'fs': fs, 'R0': R0, 'La': La}
    perf = _compute_performance(design, spec)

    # 构建元件描述
    L_desc = ", ".join(f"L{i+1}={v} nH" for i, v in enumerate(design['L']))
    C_desc = ", ".join(f"C{i+1}={v} pF" for i, v in enumerate(design['C']))

    lang = rng.choice(["zh", "en", "mix"])
    if lang == "zh":
        user_msg = (
            f"以下是一个 {filter_type} 低通滤波器（{design['N']}阶）的元件参数：\n"
            f"电感: {L_desc}\n电容: {C_desc}\n"
            f"端口阻抗: R0={R0} ohm, 截止频率: fc={fc} Hz, 阻带频率: fs={fs} Hz\n"
            f"请预测该滤波器的性能指标。"
        )
    elif lang == "en":
        user_msg = (
            f"Given a {filter_type} LPF (order {design['N']}) with:\n"
            f"Inductors: {L_desc}\nCapacitors: {C_desc}\n"
            f"R0={R0} ohm, fc={fc} Hz, fs={fs} Hz\n"
            f"Predict the performance metrics."
        )
    else:
        user_msg = (
            f"这是一个 {filter_type} LPF ({design['N']}阶):\n"
            f"Inductors: {L_desc}\nCapacitors: {C_desc}\n"
            f"R0={R0} ohm, fc={fc} Hz, fs={fs} Hz\n"
            f"请 predict 性能指标。"
        )

    assistant_msg = json.dumps(perf, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_PREDICT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def generate_comparison_sample(
    rng: random.Random,
    filter_type: str = "chebyshev",
) -> Optional[Dict]:
    """
    生成比较样本: 给定两组参数+性能 → 分析优劣

    教模型学会"哪个设计更好"的工程判断力
    """
    ripple = rng.choice([0.1, 0.2, 0.5])
    fc = rng.choice([800e6, 1e9, 1.5e9, 2e9])
    fs_ratio = rng.uniform(1.5, 2.5)
    fs = fc * fs_ratio
    La = rng.uniform(30, 50)
    R0 = 50

    designer = get_filter_designer(filter_type)

    # 设计 A: 较低阶
    order_a = rng.randint(3, 5)
    try:
        design_a = designer.design_lpf_by_attenuation(
            ripple_db=ripple, fc=fc, fs=fs, R0=R0, La=La, order=order_a
        )
    except Exception:
        return None

    # 设计 B: 较高阶
    order_b = order_a + rng.randint(1, 3)
    designer_b = get_filter_designer(filter_type)
    try:
        design_b = designer_b.design_lpf_by_attenuation(
            ripple_db=ripple, fc=fc, fs=fs, R0=R0, La=La, order=order_b
        )
    except Exception:
        return None

    spec = {'ripple_db': ripple, 'fc': fc, 'fs': fs, 'R0': R0, 'La': La}
    perf_a = _compute_performance(design_a, spec)
    perf_b = _compute_performance(design_b, spec)

    user_msg = (
        f"请比较以下两个 {filter_type} 低通滤波器设计（目标 La={La} dB）：\n\n"
        f"【设计A】阶数={design_a['N']}, 性能={json.dumps(perf_a, ensure_ascii=False)}\n"
        f"【设计B】阶数={design_b['N']}, 性能={json.dumps(perf_b, ensure_ascii=False)}\n\n"
        f"哪个设计更优？请分析。"
    )

    # 判断哪个更好
    a_pass = perf_a["stopband_attenuation_dB"] >= La
    b_pass = perf_b["stopband_attenuation_dB"] >= La

    if b_pass and not a_pass:
        winner = "B"
        reason = f"设计B的阻带衰减({perf_b['stopband_attenuation_dB']} dB)达标，设计A({perf_a['stopband_attenuation_dB']} dB)未达标。"
    elif a_pass and not b_pass:
        winner = "A"
        reason = f"设计A的阻带衰减达标，设计B未达标。"
    elif a_pass and b_pass:
        # 两者都达标，选择阶数低的（更简单、成本低）
        winner = "A"
        reason = (
            f"两者均达标。设计A阶数更低({design_a['N']}阶 vs {design_b['N']}阶)，"
            f"元件数更少，实现成本更低，群延时更小，推荐选择设计A。"
        )
    else:
        winner = "B"
        reason = (
            f"两者均未达标，但设计B的阻带衰减({perf_b['stopband_attenuation_dB']} dB)"
            f"更接近目标({La} dB)，调整空间更大。"
        )

    assistant_msg = json.dumps({
        "winner": winner,
        "analysis": reason,
        "design_A_pass": a_pass,
        "design_B_pass": b_pass,
    }, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPARE},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="性能预测/比较数据生成")
    parser.add_argument("--num-predict", type=int, default=300, help="正向预测样本数")
    parser.add_argument("--num-compare", type=int, default=100, help="比较分析样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="reflection_dataset")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 正向预测
    predict_samples = []
    for i in range(args.num_predict):
        ft = rng.choice(["chebyshev", "butterworth"])
        s = generate_forward_prediction_sample(rng, filter_type=ft)
        if s:
            predict_samples.append(s)
        if (i + 1) % 50 == 0:
            print(f"正向预测: {i+1}/{args.num_predict}")

    # 比较分析
    compare_samples = []
    for i in range(args.num_compare):
        ft = rng.choice(["chebyshev", "butterworth"])
        s = generate_comparison_sample(rng, filter_type=ft)
        if s:
            compare_samples.append(s)
        if (i + 1) % 50 == 0:
            print(f"比较分析: {i+1}/{args.num_compare}")

    # 合并输出
    all_samples = predict_samples + compare_samples
    rng.shuffle(all_samples)

    out_path = out_dir / "reverse_prediction_train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n生成完成:")
    print(f"  正向预测: {len(predict_samples)} 条")
    print(f"  比较分析: {len(compare_samples)} 条")
    print(f"  总计: {len(all_samples)} 条")
    print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
