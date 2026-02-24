r"""
直接运行:
    python make_sft_dataset_zhmix.py

如需临时覆盖参数（可选）:
    python make_sft_dataset_zhmix.py --seed 123 --augment-full 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# 配置区--·--------------
INPUT_PATH = r"G:\wenlong\ADS\filter_dataset_results_20260224\cleaned_dataset.json"
OUTPUT_DIR = r"G:\wenlong\ADS\filter_dataset_results_20260224\sft_zhmix"

# 随机种子，确保结果可复现
SEED = 42
# 每个数据项生成的全参数样本数量
AUGMENT_FULL = 4
# 生成跟进问答样本的概率（0-1之间）
FOLLOWUP_RATIO = 0.45
# 在跟进样本中，生成多轮对话的概率
RESOLVE_RATIO = 0.7
# 用户表达使用中文的比例
ZH_RATIO = 0.6
# 用户表达使用中英混合的比例
MIX_RATIO = 0.25
# 英文比例 = 1.0 - ZH_RATIO - MIX_RATIO = 0.15
# 训练集占比
TRAIN_RATIO = 0.9
# 验证集占比
VAL_RATIO = 0.05
# 测试集占比 = 1.0 - TRAIN_RATIO - VAL_RATIO = 0.05
#-----------------

SYSTEM_PROMPT = (
    "你是滤波器设计助手，支持低通(LPF)、高通(HPF)、带通(BPF)滤波器设计。"
    "当用户信息完整时，只输出一个JSON对象。"
    "低通/高通: 键必须为 filter_type, filter_band, ripple_db, fc, fs, R0, La_target, order。"
    "带通: 键必须为 filter_type, filter_band, ripple_db, f_center, bandwidth, fs_lower, fs_upper, R0, La_target, order。"
    "filter_band 取值: lowpass / highpass / bandpass。"
    "当用户缺参时，先追问缺失参数（仅追问缺失项）。"
    "绝不追问order；若用户未提供order，你需要自动补全order。"
    "单位要求: fc/fs/f_center/bandwidth 用Hz，R0用ohm，ripple_db/La_target用dB。"
)

# =================== 低通 (LPF) 模板 ===================
FULL_TEMPLATES_ZH = [
    "请设计一个{filter_type}低通滤波器：通带纹波{ripple_db} dB，截止频率{fc} Hz，阻带频率{fs} Hz，端口阻抗{R0} ohm，阻带衰减目标{La_target} dB。",
    "我要做{filter_type}低通，参数是 ripple={ripple_db} dB，fc={fc} Hz，fs={fs} Hz，R0={R0} ohm，La_target={La_target} dB。",
    "帮我给出一个{filter_type}低通滤波器规格：fc={fc} Hz，fs={fs} Hz，ripple={ripple_db} dB，R0={R0} ohm，La={La_target} dB。",
]

FULL_TEMPLATES_EN = [
    "Design a {filter_type} low-pass filter with ripple {ripple_db} dB, cutoff {fc} Hz, stopband {fs} Hz, R0 {R0} ohm, and La_target {La_target} dB.",
    "Need a {filter_type} LPF: ripple={ripple_db} dB, fc={fc} Hz, fs={fs} Hz, R0={R0} ohm, La={La_target} dB.",
]

FULL_TEMPLATES_MIX = [
    "帮我设计 {filter_type} LPF，ripple={ripple_db} dB，fc={fc} Hz，fs={fs} Hz，R0={R0} ohm，La_target={La_target} dB。",
    "Need 一个 {filter_type} 低通滤波器: fc={fc} Hz, fs={fs} Hz, ripple={ripple_db} dB, R0={R0} ohm, La={La_target} dB。",
    "请给我 {filter_type} low-pass 参数，fc {fc} Hz，fs {fs} Hz，R0 {R0} ohm，ripple {ripple_db} dB，La_target {La_target} dB。",
]

# =================== 高通 (HPF) 模板 ===================
HPF_FULL_TEMPLATES_ZH = [
    "请设计一个{filter_type}高通滤波器：通带纹波{ripple_db} dB，通带截止频率{fc} Hz，阻带频率{fs} Hz，端口阻抗{R0} ohm，阻带衰减{La_target} dB。",
    "我要做{filter_type}高通，参数: ripple={ripple_db} dB，fc={fc} Hz，fs={fs} Hz，R0={R0} ohm，La_target={La_target} dB。",
]

HPF_FULL_TEMPLATES_EN = [
    "Design a {filter_type} high-pass filter: ripple {ripple_db} dB, cutoff {fc} Hz, stopband {fs} Hz, R0 {R0} ohm, La_target {La_target} dB.",
    "Need a {filter_type} HPF: ripple={ripple_db} dB, fc={fc} Hz, fs={fs} Hz, R0={R0} ohm, La={La_target} dB.",
]

HPF_FULL_TEMPLATES_MIX = [
    "帮我设计 {filter_type} HPF 高通滤波器，ripple={ripple_db} dB，fc={fc} Hz，fs={fs} Hz，R0={R0} ohm，La_target={La_target} dB。",
    "Need {filter_type} 高通 filter: fc={fc} Hz, fs={fs} Hz, ripple={ripple_db} dB, R0={R0} ohm, La={La_target} dB。",
]

# =================== 带通 (BPF) 模板 ===================
BPF_FULL_TEMPLATES_ZH = [
    "请设计一个{filter_type}带通滤波器：中心频率{f_center} Hz，带宽{bandwidth} Hz，下阻带{fs_lower} Hz，上阻带{fs_upper} Hz，纹波{ripple_db} dB，R0={R0} ohm，La_target={La_target} dB。",
    "我要做{filter_type}带通，f_center={f_center} Hz，bandwidth={bandwidth} Hz，fs_lower={fs_lower} Hz，fs_upper={fs_upper} Hz，ripple={ripple_db} dB，R0={R0} ohm，La={La_target} dB。",
]

BPF_FULL_TEMPLATES_EN = [
    "Design a {filter_type} band-pass filter: center {f_center} Hz, BW {bandwidth} Hz, lower stopband {fs_lower} Hz, upper stopband {fs_upper} Hz, ripple {ripple_db} dB, R0 {R0} ohm, La_target {La_target} dB.",
    "Need a {filter_type} BPF: f_center={f_center} Hz, bandwidth={bandwidth} Hz, fs_lower={fs_lower} Hz, fs_upper={fs_upper} Hz, ripple={ripple_db} dB, R0={R0} ohm, La={La_target} dB.",
]

BPF_FULL_TEMPLATES_MIX = [
    "帮我设计 {filter_type} BPF 带通滤波器: f_center={f_center} Hz, bandwidth={bandwidth} Hz, fs_lower={fs_lower} Hz, fs_upper={fs_upper} Hz, ripple={ripple_db} dB, R0={R0} ohm, La_target={La_target} dB。",
    "Need {filter_type} 带通 filter: center {f_center} Hz, BW {bandwidth} Hz, lower {fs_lower} Hz, upper {fs_upper} Hz, ripple {ripple_db} dB, R0 {R0} ohm, La={La_target} dB。",
]

# =================== 通用 Partial 模板 ===================
PARTIAL_TEMPLATES_ZH = [
    "我想做滤波器，目前已知参数有：{provided_fields}。",
    "请帮我设计滤波器，我现在只有这些参数：{provided_fields}。",
]

PARTIAL_TEMPLATES_EN = [
    "I need a filter; currently I only know: {provided_fields}.",
    "Please design a filter. Known params: {provided_fields}.",
]

PARTIAL_TEMPLATES_MIX = [
    "请帮我设计 filter，我现在只知道：{provided_fields}。",
    "Need filter design，目前已知参数: {provided_fields}。",
]

FIELD_LABELS = {
    "filter_type": "filter_type",
    "filter_band": "filter_band",
    "ripple_db": "ripple_db",
    "fc": "fc",
    "fs": "fs",
    "f_center": "f_center",
    "bandwidth": "bandwidth",
    "fs_lower": "fs_lower",
    "fs_upper": "fs_upper",
    "R0": "R0",
    "La_target": "La_target",
    "order": "order",
}

REQUIRED_FIELDS_LPF = ["filter_type", "ripple_db", "fc", "fs", "R0", "La_target"]
REQUIRED_FIELDS_HPF = ["filter_type", "ripple_db", "fc", "fs", "R0", "La_target"]
REQUIRED_FIELDS_BPF = ["filter_type", "ripple_db", "f_center", "bandwidth", "fs_lower", "fs_upper", "R0", "La_target"]
# Keep backward compat alias
REQUIRED_FIELDS = REQUIRED_FIELDS_LPF


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


def extract_params(item: Dict) -> Dict[str, float]:
    design = item.get("design", {})
    params = design.get("params", {})
    order = design.get("N", params.get("order", 5))
    filter_band = design.get("filter_band", item.get("filter_band", "lowpass"))

    base = {
        "filter_type": design.get("filter_type", item.get("filter_type", "chebyshev")),
        "filter_band": filter_band,
        "ripple_db": float(params.get("ripple_db", 0.1)),
        "R0": float(params.get("R0", 50)),
        "La_target": float(params.get("La_target", 30)),
        "order": int(order),
    }

    if filter_band == "bandpass":
        base["f_center"] = float(params.get("f_center", 1e9))
        base["bandwidth"] = float(params.get("bandwidth", 2e8))
        base["fs_lower"] = float(params.get("fs_lower", 5e8))
        base["fs_upper"] = float(params.get("fs_upper", 1.5e9))
    else:
        base["fc"] = float(params.get("fc", 1e9))
        base["fs"] = float(params.get("fs", 2e9))

    return base


def assistant_json(params: Dict[str, float]) -> str:
    payload = {
        "filter_type": params["filter_type"],
        "filter_band": params.get("filter_band", "lowpass"),
        "ripple_db": float(params["ripple_db"]),
        "R0": float(params["R0"]),
        "La_target": float(params["La_target"]),
        "order": int(params["order"]),
    }
    fb = params.get("filter_band", "lowpass")
    if fb == "bandpass":
        payload["f_center"] = float(params["f_center"])
        payload["bandwidth"] = float(params["bandwidth"])
        payload["fs_lower"] = float(params["fs_lower"])
        payload["fs_upper"] = float(params["fs_upper"])
    else:
        payload["fc"] = float(params["fc"])
        payload["fs"] = float(params["fs"])
    return json.dumps(payload, ensure_ascii=True)


def target_key(params: Dict[str, float]) -> str:
    return assistant_json(params)


def render_full_user(params: Dict[str, float], template: str) -> str:
    fb = params.get("filter_band", "lowpass")
    fmt_dict = {
        "filter_type": params["filter_type"],
        "ripple_db": fmt_num(params["ripple_db"]),
        "R0": fmt_num(params["R0"]),
        "La_target": fmt_num(params["La_target"]),
    }
    if fb == "bandpass":
        fmt_dict["f_center"] = fmt_num(params["f_center"])
        fmt_dict["bandwidth"] = fmt_num(params["bandwidth"])
        fmt_dict["fs_lower"] = fmt_num(params["fs_lower"])
        fmt_dict["fs_upper"] = fmt_num(params["fs_upper"])
    else:
        fmt_dict["fc"] = fmt_num(params["fc"])
        fmt_dict["fs"] = fmt_num(params["fs"])
    return template.format(**fmt_dict)


def provided_fields_text(params: Dict[str, float], include_fields: List[str]) -> str:
    out: List[str] = []
    for key in include_fields:
        if key not in params:
            continue
        v = params[key]
        if key in ("fc", "fs", "f_center", "bandwidth", "fs_lower", "fs_upper"):
            out.append(f"{key}={fmt_num(v)} Hz")
        elif key == "R0":
            out.append(f"{key}={fmt_num(v)} ohm")
        elif key in ("ripple_db", "La_target"):
            out.append(f"{key}={fmt_num(v)} dB")
        else:
            out.append(f"{key}={v}")
    return "，".join(out)


def followup_question(missing_fields: List[str], zh: bool) -> str:
    names = [FIELD_LABELS[f] for f in missing_fields]
    if zh:
        return "请补充以下缺失参数：" + "、".join(names) + "。阶数(order)可以不提供，我会自动补全。"
    return "Please provide missing parameters: " + ", ".join(names) + ". order can be omitted and will be auto-filled."


def user_reply_missing(params: Dict[str, float], missing_fields: List[str], zh: bool) -> str:
    parts = []
    for key in missing_fields:
        if key not in params:
            continue
        if key in ("fc", "fs", "f_center", "bandwidth", "fs_lower", "fs_upper"):
            parts.append(f"{key}={fmt_num(params[key])} Hz")
        elif key == "R0":
            parts.append(f"{key}={fmt_num(params[key])} ohm")
        elif key in ("ripple_db", "La_target"):
            parts.append(f"{key}={fmt_num(params[key])} dB")
        else:
            parts.append(f"{key}={params[key]}")
    text = ", ".join(parts)
    if zh:
        return f"补充参数：{text}。"
    return f"Here are the missing values: {text}."


def choose_lang(rng: random.Random, zh_ratio: float, mix_ratio: float) -> str:
    x = rng.random()
    if x < zh_ratio:
        return "zh"
    if x < zh_ratio + mix_ratio:
        return "mix"
    return "en"


def templates_by_lang(lang: str, full: bool, filter_band: str = "lowpass") -> List[str]:
    if full:
        if filter_band == "highpass":
            return {"zh": HPF_FULL_TEMPLATES_ZH, "en": HPF_FULL_TEMPLATES_EN, "mix": HPF_FULL_TEMPLATES_MIX}[lang]
        elif filter_band == "bandpass":
            return {"zh": BPF_FULL_TEMPLATES_ZH, "en": BPF_FULL_TEMPLATES_EN, "mix": BPF_FULL_TEMPLATES_MIX}[lang]
        return {
            "zh": FULL_TEMPLATES_ZH,
            "en": FULL_TEMPLATES_EN,
            "mix": FULL_TEMPLATES_MIX,
        }[lang]
    return {
        "zh": PARTIAL_TEMPLATES_ZH,
        "en": PARTIAL_TEMPLATES_EN,
        "mix": PARTIAL_TEMPLATES_MIX,
    }[lang]


def dedup_samples(samples: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for s in samples:
        k = json.dumps(s["messages"], ensure_ascii=True, sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def split_by_group(samples: List[Dict], train_ratio: float, val_ratio: float, rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[s["_group"]].append(s)

    keys = list(groups.keys())
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    train, val, test = [], [], []
    for k, arr in groups.items():
        if k in train_keys:
            train.extend(arr)
        elif k in val_keys:
            val.extend(arr)
        elif k in test_keys:
            test.extend(arr)

    return train, val, test


def write_jsonl(path: Path, items: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            obj = {"messages": item["messages"]}
            f.write(json.dumps(obj, ensure_ascii=True))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhanced SFT dataset builder")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Path to cleaned_dataset.json")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--augment-full", type=int, default=AUGMENT_FULL, help="Full-parameter prompts per item")
    parser.add_argument("--followup-ratio", type=float, default=FOLLOWUP_RATIO, help="Probability to add follow-up sample per item")
    parser.add_argument("--resolve-ratio", type=float, default=RESOLVE_RATIO, help="Among follow-up samples, ratio for multi-turn resolved dialogs")
    parser.add_argument("--zh-ratio", type=float, default=ZH_RATIO, help="User expression zh ratio")
    parser.add_argument("--mix-ratio", type=float, default=MIX_RATIO, help="User expression mixed ratio")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    print("=" * 70)
    print("SFT 数据集构建配置摘要")
    print("=" * 70)
    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"seed: {args.seed}")
    print(f"augment_full: {args.augment_full}")
    print(f"followup_ratio: {args.followup_ratio}")
    print(f"resolve_ratio: {args.resolve_ratio}")
    print(f"zh_ratio: {args.zh_ratio}")
    print(f"mix_ratio: {args.mix_ratio}")
    print(f"en_ratio: {max(0.0, 1.0 - args.zh_ratio - args.mix_ratio)}")
    print(f"train_ratio: {args.train_ratio}")
    print(f"val_ratio: {args.val_ratio}")
    print("order_policy: never ask order, auto-fill if missing")
    print("=" * 70)

    if args.zh_ratio < 0 or args.mix_ratio < 0 or (args.zh_ratio + args.mix_ratio) > 1.0:
        raise ValueError("zh-ratio and mix-ratio must be >=0 and zh+mix <= 1.0")

    rng = random.Random(args.seed)

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input cleaned dataset must be a list")

    augment_full = max(1, int(args.augment_full))
    followup_ratio = max(0.0, min(1.0, float(args.followup_ratio)))
    resolve_ratio = max(0.0, min(1.0, float(args.resolve_ratio)))

    samples: List[Dict] = []
    type_counter = Counter()
    lang_counter = Counter()

    for item in data:
        params = extract_params(item)
        gkey = target_key(params)
        filter_band = params.get("filter_band", "lowpass")

        # Determine required fields based on filter band
        if filter_band == "bandpass":
            req_fields = REQUIRED_FIELDS_BPF
        elif filter_band == "highpass":
            req_fields = REQUIRED_FIELDS_HPF
        else:
            req_fields = REQUIRED_FIELDS_LPF

        # Full samples
        for _ in range(augment_full):
            lang = choose_lang(rng, args.zh_ratio, args.mix_ratio)
            tpl = rng.choice(templates_by_lang(lang, full=True, filter_band=filter_band))
            user = render_full_user(params, tpl)
            samples.append(
                {
                    "_group": gkey,
                    "_type": "full",
                    "_lang": lang,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant_json(params)},
                    ],
                }
            )
            type_counter["full"] += 1
            lang_counter[lang] += 1

        # Follow-up samples
        if rng.random() < followup_ratio:
            omit_count = rng.randint(1, min(3, len(req_fields) - 1))
            omitted = rng.sample(req_fields, k=omit_count)
            included = [k for k in req_fields if k not in omitted]

            lang = choose_lang(rng, args.zh_ratio, args.mix_ratio)
            partial_tpl = rng.choice(templates_by_lang(lang, full=False, filter_band=filter_band))
            partial_user = partial_tpl.format(provided_fields=provided_fields_text(params, included))
            ask = followup_question(omitted, zh=(lang != "en"))

            if rng.random() < resolve_ratio:
                # Multi-turn resolved dialog
                user2 = user_reply_missing(params, omitted, zh=(lang != "en"))
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": partial_user},
                    {"role": "assistant", "content": ask},
                    {"role": "user", "content": user2},
                    {"role": "assistant", "content": assistant_json(params)},
                ]
                stype = "followup_resolve"
            else:
                # Single-turn ask only
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": partial_user},
                    {"role": "assistant", "content": ask},
                ]
                stype = "followup_question"

            samples.append(
                {
                    "_group": gkey,
                    "_type": stype,
                    "_lang": lang,
                    "messages": messages,
                }
            )
            type_counter[stype] += 1
            lang_counter[lang] += 1

    samples = dedup_samples(samples)

    final_type_counter = Counter(s["_type"] for s in samples)
    final_lang_counter = Counter(s["_lang"] for s in samples)

    train_set, val_set, test_set = split_by_group(samples, args.train_ratio, args.val_ratio, rng)
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    write_jsonl(out_dir / "train.jsonl", train_set)
    write_jsonl(out_dir / "val.jsonl", val_set)
    write_jsonl(out_dir / "test.jsonl", test_set)

    meta = {
        "input_items": len(data),
        "output_samples": len(samples),
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "type_distribution": dict(final_type_counter),
        "lang_distribution": dict(final_lang_counter),
        "config": {
            "augment_full": augment_full,
            "followup_ratio": followup_ratio,
            "resolve_ratio": resolve_ratio,
            "zh_ratio": args.zh_ratio,
            "mix_ratio": args.mix_ratio,
            "en_ratio": max(0.0, 1.0 - args.zh_ratio - args.mix_ratio),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "order_policy": "never ask order, auto-fill if missing",
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"数据载入: {len(data)}")
    print(f"产生微调样本: {len(samples)}")
    print(f"训练集/验证集/测试集: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"类型分布: {dict(final_type_counter)}")
    print(f"语言分布: {dict(final_lang_counter)}")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
