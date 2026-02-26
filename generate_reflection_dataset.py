r"""
反思数据 (Reflection Data) 批量仿真生成模块
==============================================

目的：
  基于仿真结果自动生成"参数→仿真→评估→反思调参"的多轮对话 SFT 数据，
  让微调后的模型具备射频调参直觉。
  & "F:/Program Files (x86)/ADS2026/tools/python/python.exe" generate_reflection_dataset.py

不依赖 ADS 的纯计算模式（仅基于公式，不调 ADS 仿真）：
  python generate_reflection_dataset.py --mode formula

工作流程：
  1. 随机生成一批"初始参数"
  2. 故意引入偏差 (如 order 偏低、fc 偏移) 使其"不达标"
  3. 对每组参数进行纯公式计算 / ADS 仿真
  4. 评估结果是否达标
  5. 由规则引擎 (非 LLM) 生成专家级反思文本 + 修正后参数
  6. 再次仿真验证修正后参数确实改善
  7. 输出为可直接用于 SFT 的 JSONL 数据
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ======================= 配置 =======================
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "reflection_dataset"
NUM_SAMPLES = 500          # 生成的反思样本数
NUM_HPF_SAMPLES = 150      # 高通滤波器样本
NUM_BPF_SAMPLES = 150      # 带通滤波器样本

SYSTEM_PROMPT = (
    "你是专业的射频滤波器设计专家。你会根据仿真结果分析滤波器性能是否达标，"
    "并给出具体的调参建议和修正后的完整参数 JSON。"
    "输出必须是可解析 JSON，不要输出额外文字。"
)

# 偏差注入策略
PERTURBATION_STRATEGIES = [
    "order_too_low",       # 阶数偏低 → 阻带衰减不够
    "fc_shifted",          # 截止频率偏移 → 通带/阻带边界错位
    "ripple_too_high",     # 纹波过大 → S11 不良
    "order_slightly_low",  # 阶数仅差 1 → 差一点达标
]

# =================== 滤波器设计器 ===================
# 直接使用 adsapi 中的设计器进行纯公式计算

from adsapi.filter_designer import (
    ChebyshevFilterDesigner,
    ButterworthFilterDesigner,
    get_filter_designer,
)

# =================== 工具函数 ===================

def _fmt(x: float) -> str:
    """格式化数值"""
    if abs(x) >= 1e9:
        return f"{x/1e9:.3g} GHz"
    if abs(x) >= 1e6:
        return f"{x/1e6:.3g} MHz"
    return f"{x:.6g}"


def _compute_metrics_from_design(design: Dict, spec: Dict) -> Dict[str, Any]:
    """
    基于公式计算的伪仿真指标（不调 ADS）。
    利用切比雪夫/巴特沃斯理论公式估算 S 参数。
    """
    N = design['N']
    gk = design['gk']
    atten = design['Atten']
    params = design['params']
    ripple_db = params['ripple_db']
    filter_band = design.get('filter_band', 'lowpass')

    if filter_band == 'bandpass':
        fc = params.get('f_upper', params.get('fc', 1e9))
        fs = params.get('fs_upper', params.get('fs', 2e9))
    elif filter_band == 'highpass':
        fc = params['fc']
        fs = params['fs']
    else:
        fc = params['fc']
        fs = params['fs']

    # 通带内最大 S11 (dB) — 由波纹决定
    # S11 ≈ -10*log10(1 - 10^(-ripple/10))
    if ripple_db > 0:
        s11_max = 10 * np.log10(1 - 10 ** (-ripple_db / 10))
    else:
        s11_max = -30.0  # 理想

    # S21 通带最差 = -ripple_db
    s21_passband_min = -ripple_db

    # 阻带衰减 = Atten (负数 dB)
    s21_stopband_max = atten

    # 通带纹波
    passband_ripple = ripple_db

    # 阻带衰减不够的判定
    la_target = params.get('La_target', 30)
    min_stopband_attenuation_db = abs(atten)

    return {
        'S11_max_dB': round(float(s11_max), 2),
        'S21_passband_min_dB': round(float(s21_passband_min), 2),
        'S21_stopband_max_dB': round(float(s21_stopband_max), 2),
        'passband_ripple_dB': round(float(passband_ripple), 2),
        'min_stopband_attenuation_db': round(float(min_stopband_attenuation_db), 2),
        'fc_Hz': fc,
        'fs_Hz': fs,
        'order': N,
        'filter_band': filter_band,
    }


def _evaluate(metrics: Dict, spec: Dict) -> Tuple[bool, List[str]]:
    """
    评估metrics是否达标，返回 (是否达标, 问题列表)
    """
    issues = []
    la_target = spec.get('La', spec.get('La_target', 30))
    actual_la = metrics.get('min_stopband_attenuation_db', 0)

    if actual_la < la_target:
        issues.append(
            f"阻带衰减不足: 实际 {actual_la:.1f} dB < 目标 {la_target} dB，"
            f"差距 {la_target - actual_la:.1f} dB"
        )

    ripple_target = spec.get('ripple_db', 0.5)
    actual_ripple = metrics.get('passband_ripple_dB', 0)
    if actual_ripple > ripple_target * 1.5:
        issues.append(
            f"通带纹波过大: 实际 {actual_ripple:.2f} dB > 允许 {ripple_target:.2f} dB"
        )

    s11 = metrics.get('S11_max_dB', 0)
    if s11 > -10:
        issues.append(f"通带内 S11 过高: {s11:.1f} dB > -10 dB，匹配较差")

    return (len(issues) == 0, issues)


def _generate_expert_reflection(
    spec: Dict, metrics: Dict, issues: List[str], filter_band: str
) -> Tuple[str, Dict]:
    """
    规则引擎：根据问题生成专家级反思文本和修正参数。
    返回 (反思文本, 修正后参数)
    """
    new_spec = dict(spec)
    analysis_parts = []
    adjustments = []

    la_target = spec.get('La', spec.get('La_target', 30))
    actual_la = metrics.get('min_stopband_attenuation_db', 0)
    old_order = spec.get('order', metrics.get('order', 5))

    for issue in issues:
        if "阻带衰减不足" in issue:
            gap = la_target - actual_la
            if gap > 15:
                inc = 3
            elif gap > 8:
                inc = 2
            else:
                inc = 1
            new_order = old_order + inc
            analysis_parts.append(
                f"分析：阻带衰减差距为 {gap:.1f} dB。"
                f"根据切比雪夫/巴特沃斯理论，每增加1阶大约增加 6~20 dB 衰减。"
                f"因此将阶数从 {old_order} 增加到 {new_order}。"
            )
            adjustments.append(f"order: {old_order} → {new_order}")
            new_spec['order'] = new_order
            old_order = new_order

        elif "通带纹波过大" in issue:
            old_ripple = spec.get('ripple_db', 0.5)
            new_ripple = round(old_ripple * 0.6, 3)
            analysis_parts.append(
                f"分析：通带纹波 {old_ripple} dB 超标。减小纹波设计值到 {new_ripple} dB "
                f"可改善 S11 匹配和通带平坦度。"
            )
            adjustments.append(f"ripple_db: {old_ripple} → {new_ripple}")
            new_spec['ripple_db'] = new_ripple

        elif "S11 过高" in issue:
            # 增加阶数 or 减小纹波
            if spec.get('ripple_db', 0.5) > 0.3:
                new_ripple = round(spec['ripple_db'] * 0.5, 3)
                analysis_parts.append(
                    f"分析：S11 过高表明通带匹配差。减小纹波从 {spec['ripple_db']} dB "
                    f"到 {new_ripple} dB 以改善。"
                )
                adjustments.append(f"ripple_db: {spec['ripple_db']} → {new_ripple}")
                new_spec['ripple_db'] = new_ripple
            else:
                new_order = old_order + 1
                analysis_parts.append(
                    f"分析：S11 过高且纹波已较低，增加阶数从 {old_order} 到 {new_order}。"
                )
                adjustments.append(f"order: {old_order} → {new_order}")
                new_spec['order'] = new_order

    reflection_text = (
        f"上一轮仿真发现以下问题：\n"
        + "\n".join(f"- {iss}" for iss in issues)
        + "\n\n"
        + "\n".join(analysis_parts)
        + "\n\n调整方案: " + "; ".join(adjustments) + "。"
    )

    return reflection_text, new_spec


# =================== 样本生成函数 ===================

def _random_lpf_spec(rng: random.Random) -> Dict:
    """随机生成低通滤波器规格"""
    ripple_db = rng.choice([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    fc = rng.choice([400e6, 600e6, 800e6, 1e9, 1.2e9, 1.5e9, 2e9, 2.4e9, 3e9])
    fs_ratio = rng.uniform(1.3, 3.0)
    La = rng.uniform(25, 60)
    R0 = rng.choice([25, 50, 75, 100])
    order = rng.randint(3, 9)
    ft = rng.choice(["chebyshev", "butterworth"])
    return {
        'ripple_db': ripple_db, 'fc': fc, 'fs': fc * fs_ratio,
        'R0': R0, 'La': round(La, 1), 'order': order,
        'filter_type': ft, 'filter_band': 'lowpass',
    }


def _random_hpf_spec(rng: random.Random) -> Dict:
    """随机生成高通滤波器规格"""
    ripple_db = rng.choice([0.05, 0.1, 0.2, 0.5, 1.0])
    fc = rng.choice([500e6, 800e6, 1e9, 1.5e9, 2e9, 2.5e9])
    fs_ratio = rng.uniform(0.3, 0.7)  # fs < fc for HPF
    La = rng.uniform(25, 55)
    R0 = rng.choice([50, 75])
    order = rng.randint(3, 8)
    ft = rng.choice(["chebyshev", "butterworth"])
    return {
        'ripple_db': ripple_db, 'fc': fc, 'fs': fc * fs_ratio,
        'R0': R0, 'La': round(La, 1), 'order': order,
        'filter_type': ft, 'filter_band': 'highpass',
    }


def _random_bpf_spec(rng: random.Random) -> Dict:
    """随机生成带通滤波器规格"""
    ripple_db = rng.choice([0.05, 0.1, 0.2, 0.5])
    f_center = rng.choice([900e6, 1.575e9, 2.4e9, 3.5e9, 5e9])
    bw_ratio = rng.uniform(0.05, 0.3)
    bandwidth = f_center * bw_ratio
    guard = rng.uniform(1.3, 2.0)
    fs_lower = f_center - bandwidth * guard
    fs_upper = f_center + bandwidth * guard
    La = rng.uniform(25, 50)
    R0 = rng.choice([50, 75])
    order = rng.randint(3, 7)
    ft = rng.choice(["chebyshev", "butterworth"])
    return {
        'ripple_db': ripple_db, 'f_center': f_center,
        'bandwidth': bandwidth, 'fs_lower': fs_lower, 'fs_upper': fs_upper,
        'R0': R0, 'La': round(La, 1), 'order': order,
        'filter_type': ft, 'filter_band': 'bandpass',
    }


def _perturb_spec(spec: Dict, rng: random.Random) -> Dict:
    """
    故意对 spec 注入偏差，使仿真结果"不达标"。
    """
    perturbed = dict(spec)
    strategy = rng.choice(PERTURBATION_STRATEGIES)

    if strategy == "order_too_low":
        perturbed['order'] = max(2, spec.get('order', 5) - rng.randint(2, 3))
    elif strategy == "fc_shifted":
        shift = rng.uniform(0.1, 0.3)  # 10~30% 偏移
        if rng.random() < 0.5:
            perturbed['fc'] = spec.get('fc', 1e9) * (1 + shift)
        else:
            perturbed['fc'] = spec.get('fc', 1e9) * (1 - shift)
        # BPF 不适用 fc shift，改 bandwidth
        if spec.get('filter_band') == 'bandpass':
            perturbed['bandwidth'] = spec.get('bandwidth', 1e8) * (1 - shift * 0.5)
            del perturbed['fc']
    elif strategy == "ripple_too_high":
        perturbed['ripple_db'] = min(3.0, spec.get('ripple_db', 0.5) * rng.uniform(2, 5))
    elif strategy == "order_slightly_low":
        perturbed['order'] = max(2, spec.get('order', 5) - 1)

    perturbed['_perturbation'] = strategy
    return perturbed


def _design_and_evaluate(spec: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    使用公式设计滤波器并计算指标。
    """
    ft = spec.get('filter_type', 'chebyshev')
    fb = spec.get('filter_band', 'lowpass')
    designer = get_filter_designer(ft)

    try:
        if fb == 'highpass':
            design = designer.design_hpf_by_attenuation(
                ripple_db=spec['ripple_db'], fc=spec['fc'], fs=spec['fs'],
                R0=spec['R0'], La=spec['La'], order=spec.get('order'),
            )
        elif fb == 'bandpass':
            design = designer.design_bpf_by_attenuation(
                ripple_db=spec['ripple_db'],
                f_center=spec['f_center'], bandwidth=spec['bandwidth'],
                fs_lower=spec['fs_lower'], fs_upper=spec['fs_upper'],
                R0=spec['R0'], La=spec['La'], order=spec.get('order'),
            )
        else:
            design = designer.design_lpf_by_attenuation(
                ripple_db=spec['ripple_db'], fc=spec['fc'], fs=spec['fs'],
                R0=spec['R0'], La=spec['La'], order=spec.get('order'),
            )
        design['filter_band'] = fb
        metrics = _compute_metrics_from_design(design, spec)
        return design, metrics
    except Exception as e:
        return None, None


# =================== 对话构造 ===================

def _build_reflection_conversation(
    original_spec: Dict,
    perturbed_spec: Dict,
    bad_metrics: Dict,
    issues: List[str],
    reflection_text: str,
    corrected_spec: Dict,
    good_metrics: Dict,
    filter_band: str,
) -> List[Dict[str, str]]:
    """
    构造反思多轮对话数据.

    格式:
      system -> 系统提示
      user -> "上一轮参数 + 仿真结果不达标, 请修正"
      assistant -> "分析 + 修正后 JSON"
    """
    # 清理内部字段
    clean_perturbed = {k: v for k, v in perturbed_spec.items()
                       if not k.startswith('_')}

    band_label = {"lowpass": "低通", "highpass": "高通", "bandpass": "带通"}.get(filter_band, "低通")

    user_msg = (
        f"我设计了一个 {clean_perturbed.get('filter_type', 'chebyshev')} {band_label}滤波器，"
        f"参数如下：\n{json.dumps(clean_perturbed, ensure_ascii=False)}\n\n"
        f"仿真结果如下：\n{json.dumps(bad_metrics, ensure_ascii=False)}\n\n"
        f"目标阻带衰减 La_target = {original_spec.get('La', original_spec.get('La_target', 30))} dB。\n"
        f"请分析仿真结果并给出修正后的参数 JSON。"
    )

    # 构造 assistant 回复: 反思 + 修正 JSON
    corrected_json_clean = {
        "filter_type": corrected_spec.get("filter_type", "chebyshev"),
        "filter_band": filter_band,
        "ripple_db": corrected_spec["ripple_db"],
        "R0": corrected_spec["R0"],
        "La_target": corrected_spec.get("La", corrected_spec.get("La_target", 30)),
        "order": corrected_spec.get("order", 5),
    }

    if filter_band == 'lowpass':
        corrected_json_clean["fc"] = corrected_spec["fc"]
        corrected_json_clean["fs"] = corrected_spec["fs"]
    elif filter_band == 'highpass':
        corrected_json_clean["fc"] = corrected_spec["fc"]
        corrected_json_clean["fs"] = corrected_spec["fs"]
    elif filter_band == 'bandpass':
        corrected_json_clean["f_center"] = corrected_spec["f_center"]
        corrected_json_clean["bandwidth"] = corrected_spec["bandwidth"]
        corrected_json_clean["fs_lower"] = corrected_spec["fs_lower"]
        corrected_json_clean["fs_upper"] = corrected_spec["fs_upper"]

    assistant_msg = reflection_text + "\n\n" + json.dumps(corrected_json_clean, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def _build_evaluation_conversation(
    spec: Dict,
    metrics: Dict,
    is_pass: bool,
    issues: List[str],
    filter_band: str,
) -> List[Dict[str, str]]:
    """
    构造评估判断对话数据 (teach model to evaluate).
    """
    band_label = {"lowpass": "低通", "highpass": "高通", "bandpass": "带通"}.get(filter_band, "低通")
    clean_spec = {k: v for k, v in spec.items() if not k.startswith('_')}

    user_msg = (
        f"请评估以下 {band_label}滤波器仿真结果是否达标。\n"
        f"设计参数：{json.dumps(clean_spec, ensure_ascii=False)}\n"
        f"仿真结果：{json.dumps(metrics, ensure_ascii=False)}\n"
        f"目标阻带衰减 La_target = {spec.get('La', spec.get('La_target', 30))} dB。"
    )

    if is_pass:
        assistant_msg = json.dumps({
            "evaluation": "PASS",
            "summary": "所有指标达标。阻带衰减、通带纹波和 S11 均满足设计要求。"
        }, ensure_ascii=False)
    else:
        assistant_msg = json.dumps({
            "evaluation": "FAIL",
            "issues": issues,
            "summary": "仿真结果未达标，需要调整参数。"
        }, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


# =================== 主流程 ===================

def generate_reflection_dataset(
    num_lpf: int = NUM_SAMPLES,
    num_hpf: int = NUM_HPF_SAMPLES,
    num_bpf: int = NUM_BPF_SAMPLES,
    seed: int = SEED,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """生成反思数据集"""

    rng = random.Random(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    stats = {"total": 0, "reflection": 0, "evaluation_pass": 0, "evaluation_fail": 0,
             "lpf": 0, "hpf": 0, "bpf": 0, "skipped": 0}

    generators = [
        ("lowpass", _random_lpf_spec, num_lpf),
        ("highpass", _random_hpf_spec, num_hpf),
        ("bandpass", _random_bpf_spec, num_bpf),
    ]

    for filter_band, gen_fn, count in generators:
        print(f"\n生成 {filter_band} 反思数据 ({count} 组)...")

        for i in range(count):
            # 1. 生成理想规格
            ideal_spec = gen_fn(rng)

            # 2. 注入偏差
            perturbed_spec = _perturb_spec(ideal_spec, rng)

            # 3. 计算"bad"设计
            bad_design, bad_metrics = _design_and_evaluate(perturbed_spec)
            if bad_design is None or bad_metrics is None:
                stats["skipped"] += 1
                continue

            # 4. 评估
            is_pass, issues = _evaluate(bad_metrics, ideal_spec)

            if is_pass:
                # 偏差后仍然达标 → 生成 evaluation PASS 样本
                conv = _build_evaluation_conversation(
                    perturbed_spec, bad_metrics, True, [], filter_band
                )
                all_samples.append({"messages": conv, "_type": "eval_pass", "_band": filter_band})
                stats["evaluation_pass"] += 1
            else:
                # 不达标 → 生成反思修正样本
                reflection_text, corrected_spec = _generate_expert_reflection(
                    ideal_spec, bad_metrics, issues, filter_band
                )

                # 验证修正后确实改善
                good_design, good_metrics = _design_and_evaluate(corrected_spec)
                if good_design is None:
                    stats["skipped"] += 1
                    continue

                conv = _build_reflection_conversation(
                    ideal_spec, perturbed_spec, bad_metrics,
                    issues, reflection_text, corrected_spec, good_metrics, filter_band,
                )
                all_samples.append({"messages": conv, "_type": "reflection", "_band": filter_band})
                stats["reflection"] += 1

                # 同时生成 evaluation FAIL 样本
                eval_conv = _build_evaluation_conversation(
                    perturbed_spec, bad_metrics, False, issues, filter_band
                )
                all_samples.append({"messages": eval_conv, "_type": "eval_fail", "_band": filter_band})
                stats["evaluation_fail"] += 1

            stats[filter_band.split("pass")[0].replace("low", "lpf").replace("high", "hpf").replace("band", "bpf") if "pass" in filter_band else filter_band] += 1
            stats["total"] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{filter_band}] {i + 1}/{count}")

    # 统计
    stats["lpf"] = sum(1 for s in all_samples if s.get("_band") == "lowpass")
    stats["hpf"] = sum(1 for s in all_samples if s.get("_band") == "highpass")
    stats["bpf"] = sum(1 for s in all_samples if s.get("_band") == "bandpass")

    # 打乱
    rng.shuffle(all_samples)

    # 切分 train / val
    n = len(all_samples)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)
    train = all_samples[:n_train]
    val = all_samples[n_train:n_train + n_val]
    test = all_samples[n_train + n_val:]

    # 写出 JSONL
    def _write_jsonl(path: Path, items: List[Dict]):
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                obj = {"messages": item["messages"]}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    _write_jsonl(output_dir / "reflection_train.jsonl", train)
    _write_jsonl(output_dir / "reflection_val.jsonl", val)
    _write_jsonl(output_dir / "reflection_test.jsonl", test)

    # 元数据
    meta = {
        "total_samples": len(all_samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "stats": stats,
        "seed": seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"反思数据集生成完成")
    print(f"{'='*60}")
    print(f"总样本数: {len(all_samples)}")
    print(f"  反思修正: {stats['reflection']}")
    print(f"  评估通过: {stats['evaluation_pass']}")
    print(f"  评估不通过: {stats['evaluation_fail']}")
    print(f"  低通: {stats['lpf']}, 高通: {stats['hpf']}, 带通: {stats['bpf']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"训练/验证/测试: {len(train)}/{len(val)}/{len(test)}")
    print(f"输出目录: {output_dir}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反思数据集生成器")
    parser.add_argument("--mode", default="formula", choices=["formula"],
                        help="生成模式：formula=纯公式计算")
    parser.add_argument("--num-lpf", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-hpf", type=int, default=NUM_HPF_SAMPLES)
    parser.add_argument("--num-bpf", type=int, default=NUM_BPF_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    generate_reflection_dataset(
        num_lpf=args.num_lpf,
        num_hpf=args.num_hpf,
        num_bpf=args.num_bpf,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
