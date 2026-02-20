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
INPUT_PATH = r"G:\wenlong\ADS\filter_dataset_results_20260212\cleaned_dataset.json"
OUTPUT_DIR = r"G:\wenlong\ADS\filter_dataset_results_20260212\sft_zhmix"

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
    "你是滤波器设计助手。"
    "当用户信息完整时，只输出一个JSON对象，键必须为: "
    "filter_type, ripple_db, fc, fs, R0, La_target, order。"
    "当用户缺参时，先追问缺失参数（仅追问缺失项）。"
    "绝不追问order；若用户未提供order，你需要自动补全order。"
    "单位要求: fc/fs 用Hz，R0用ohm，ripple_db/La_target用dB。"
)

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

PARTIAL_TEMPLATES_ZH = [
    "我想做低通滤波器，目前已知参数有：{provided_fields}。",
    "请帮我设计滤波器，我现在只有这些参数：{provided_fields}。",
]

PARTIAL_TEMPLATES_EN = [
    "I need a low-pass filter; currently I only know: {provided_fields}.",
    "Please design a filter. Known params: {provided_fields}.",
]

PARTIAL_TEMPLATES_MIX = [
    "请帮我设计 low-pass filter，我现在只知道：{provided_fields}。",
    "Need filter design，目前已知参数: {provided_fields}。",
]

FIELD_LABELS = {
    "filter_type": "filter_type",
    "ripple_db": "ripple_db",
    "fc": "fc",
    "fs": "fs",
    "R0": "R0",
    "La_target": "La_target",
    "order": "order",
}

REQUIRED_FIELDS = ["filter_type", "ripple_db", "fc", "fs", "R0", "La_target"]


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


def extract_params(item: Dict) -> Dict[str, float]:
    design = item.get("design", {})
    params = design.get("params", {})
    order = design.get("N", params.get("order", 5))
    return {
        "filter_type": design.get("filter_type", "chebyshev"),
        "ripple_db": float(params.get("ripple_db", 0.1)),
        "fc": float(params.get("fc", 1e9)),
        "fs": float(params.get("fs", 2e9)),
        "R0": float(params.get("R0", 50)),
        "La_target": float(params.get("La_target", 30)),
        "order": int(order),
    }


def assistant_json(params: Dict[str, float]) -> str:
    payload = {
        "filter_type": params["filter_type"],
        "ripple_db": float(params["ripple_db"]),
        "fc": float(params["fc"]),
        "fs": float(params["fs"]),
        "R0": float(params["R0"]),
        "La_target": float(params["La_target"]),
        "order": int(params["order"]),
    }
    return json.dumps(payload, ensure_ascii=True)


def target_key(params: Dict[str, float]) -> str:
    return assistant_json(params)


def render_full_user(params: Dict[str, float], template: str) -> str:
    return template.format(
        filter_type=params["filter_type"],
        ripple_db=fmt_num(params["ripple_db"]),
        fc=fmt_num(params["fc"]),
        fs=fmt_num(params["fs"]),
        R0=fmt_num(params["R0"]),
        La_target=fmt_num(params["La_target"]),
    )


def provided_fields_text(params: Dict[str, float], include_fields: List[str]) -> str:
    out: List[str] = []
    for key in include_fields:
        v = params[key]
        if key in ("fc", "fs"):
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
        if key in ("fc", "fs"):
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


def templates_by_lang(lang: str, full: bool) -> List[str]:
    if full:
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

        # Full samples
        for _ in range(augment_full):
            lang = choose_lang(rng, args.zh_ratio, args.mix_ratio)
            tpl = rng.choice(templates_by_lang(lang, full=True))
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
            omit_count = rng.randint(1, min(3, len(REQUIRED_FIELDS) - 1))
            omitted = rng.sample(REQUIRED_FIELDS, k=omit_count)
            included = [k for k in REQUIRED_FIELDS if k not in omitted]

            lang = choose_lang(rng, args.zh_ratio, args.mix_ratio)
            partial_tpl = rng.choice(templates_by_lang(lang, full=False))
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
    print("阶数策略: 从不询问阶数；缺失时自动补全阶数。")
    print(f"类型分布: {dict(final_type_counter)}")
    print(f"语言分布: {dict(final_lang_counter)}")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
