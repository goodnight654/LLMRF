"""
基线对比评估：比较 原始Qwen3-8B (未微调) vs Qwen3-8B-RF (微调后合并模型)
用于消融实验论文写作

用法:
    conda activate llmads
    cd G:\wenlong\llmrf
    python eval_baseline_comparison.py --n 100
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 路径 ──────────────────────────────────────────────────────────────────
BASELINE_MODEL = r"G:\wenlong\models\Qwen3-8B"
FINETUNED_MODEL = r"G:\wenlong\models\Qwen3-8B-RF"
DEFAULT_TEST = r"G:\wenlong\llmrf\LLaMA-Factory\data\filter_sft_zhmix\test.jsonl"

SYSTEM_PROMPT = (
    "你是滤波器设计助手，支持低通(LPF)、高通(HPF)、带通(BPF)滤波器设计。"
    "当用户信息完整时，只输出一个JSON对象。"
    "低通/高通: 键必须为 filter_type, filter_band, ripple_db, fc, fs, R0, La_target, order。"
    "带通: 键必须为 filter_type, filter_band, ripple_db, f_center, bandwidth, fs_lower, fs_upper, R0, La_target, order。"
    "filter_band 取值: lowpass / highpass / bandpass。"
    "当用户缺参时，先追问缺失参数（仅追问缺失项）。绝不追问order；若用户未提供order，你需要自动补全order。"
    "单位要求: fc/fs/f_center/bandwidth 用Hz，R0用ohm，ripple_db/La_target用dB。"
)

NUM_FIELDS_LPF_HPF = ["fc", "fs", "ripple_db", "R0", "La_target"]
NUM_FIELDS_BPF = ["f_center", "bandwidth", "fs_lower", "fs_upper", "ripple_db", "R0", "La_target"]
REQUIRED_KEYS_LPF_HPF = {"filter_type", "filter_band", "ripple_db", "fc", "fs", "R0", "La_target", "order"}
REQUIRED_KEYS_BPF = {"filter_type", "filter_band", "ripple_db", "f_center", "bandwidth", "fs_lower", "fs_upper", "R0", "La_target", "order"}


def classify_sample(msgs):
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    if len(assistant_msgs) == 0:
        return "invalid"
    first_asst = assistant_msgs[0]["content"]
    try:
        json.loads(first_asst)
        return "full"
    except Exception:
        pass
    if len(assistant_msgs) == 1:
        return "followup_question"
    if len(assistant_msgs) >= 2:
        return "followup_resolve"
    return "unknown"


def get_final_gt(msgs):
    for m in reversed(msgs):
        if m["role"] == "assistant":
            try:
                return json.loads(m["content"])
            except:
                return None
    return None


def rel_err(pred, gt):
    if gt == 0:
        return abs(pred - gt)
    return abs(pred - gt) / abs(gt)


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None


def load_model(model_path):
    print(f"  [加载] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  [加载完成]")
    return model, tokenizer


def run_inference(model, tokenizer, messages, max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, samples, model_name):
    """评估单个模型，返回指标字典"""
    metrics = {
        "name": model_name,
        "json_ok": 0,
        "filter_type_ok": 0,
        "filter_band_ok": 0,
        "order_ok": 0,
        "keys_complete": 0,
        "total": 0,
        "num_errs": defaultdict(list),
        "by_band": defaultdict(lambda: {"total": 0, "json_ok": 0, "order_ok": 0}),
        "followup_correct": 0,
        "followup_total": 0,
        "raw_outputs": [],
    }

    t0 = time.time()
    for idx, (stype, sample) in enumerate(samples):
        msgs = sample["messages"]

        if stype == "full":
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            gt = get_final_gt(msgs)
            if gt is None:
                continue

            pred_text = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            pred = extract_json(pred_text)
            metrics["total"] += 1
            band = gt.get("filter_band", "lowpass")
            metrics["by_band"][band]["total"] += 1

            if pred is not None:
                metrics["json_ok"] += 1
                metrics["by_band"][band]["json_ok"] += 1

                if pred.get("filter_type") == gt.get("filter_type"):
                    metrics["filter_type_ok"] += 1
                if pred.get("filter_band") == gt.get("filter_band"):
                    metrics["filter_band_ok"] += 1
                if pred.get("order") == gt.get("order"):
                    metrics["order_ok"] += 1
                    metrics["by_band"][band]["order_ok"] += 1

                # 键完整性
                req_keys = REQUIRED_KEYS_BPF if band == "bandpass" else REQUIRED_KEYS_LPF_HPF
                if req_keys.issubset(set(pred.keys())):
                    metrics["keys_complete"] += 1

                # 数值误差
                num_fields = NUM_FIELDS_BPF if band == "bandpass" else NUM_FIELDS_LPF_HPF
                for f in num_fields:
                    gt_v = gt.get(f)
                    pred_v = pred.get(f)
                    if gt_v is not None and pred_v is not None:
                        try:
                            metrics["num_errs"][f].append(rel_err(float(pred_v), float(gt_v)))
                        except:
                            pass

            metrics["raw_outputs"].append({
                "idx": idx, "type": stype, "user": user_msg[:80],
                "gt_band": band, "gt_order": gt.get("order"),
                "pred_raw": pred_text[:200], "pred_json": pred,
                "json_ok": pred is not None,
            })

        elif stype in ("followup_question", "followup_resolve"):
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            pred_text = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            metrics["followup_total"] += 1
            # 检查是否正确地追问（而非直接输出JSON）
            pred_json = extract_json(pred_text)
            if pred_json is None:  # 没有输出JSON = 正确追问
                keywords = ["请补充", "请提供", "缺失", "missing", "provide", "补充", "还需要", "缺少"]
                if any(k in pred_text for k in keywords):
                    metrics["followup_correct"] += 1

            metrics["raw_outputs"].append({
                "idx": idx, "type": stype, "user": user_msg[:80],
                "pred_raw": pred_text[:200],
                "correctly_asked": pred_json is None,
            })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(samples) - idx - 1)
            print(f"    [{idx+1}/{len(samples)}] elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    metrics["total_time"] = time.time() - t0
    return metrics


def print_comparison(m_base, m_ft):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print(f"{'模型对比评估结果':^80}")
    print("=" * 80)

    header = f"{'指标':30s} | {'原始Qwen3-8B':>18s} | {'Qwen3-8B-RF':>18s}"
    print(header)
    print("-" * 80)

    def pct(v, t):
        if t == 0:
            return "N/A"
        return f"{v/t*100:.1f}% ({v}/{t})"

    rows = [
        ("评估样本数 (Full)", str(m_base["total"]), str(m_ft["total"])),
        ("JSON解析成功率", pct(m_base["json_ok"], m_base["total"]), pct(m_ft["json_ok"], m_ft["total"])),
        ("filter_type 准确率", pct(m_base["filter_type_ok"], m_base["total"]), pct(m_ft["filter_type_ok"], m_ft["total"])),
        ("filter_band 准确率", pct(m_base["filter_band_ok"], m_base["total"]), pct(m_ft["filter_band_ok"], m_ft["total"])),
        ("order 准确率", pct(m_base["order_ok"], m_base["total"]), pct(m_ft["order_ok"], m_ft["total"])),
        ("键完整率", pct(m_base["keys_complete"], m_base["total"]), pct(m_ft["keys_complete"], m_ft["total"])),
    ]
    if m_base["followup_total"] > 0:
        rows.append(("追问正确率", pct(m_base["followup_correct"], m_base["followup_total"]),
                      pct(m_ft["followup_correct"], m_ft["followup_total"])))

    for label, v1, v2 in rows:
        print(f"  {label:30s} | {v1:>18s} | {v2:>18s}")

    # 分频带对比
    print(f"\n  {'按频带 order 准确率':30s} | {'原始Qwen3-8B':>18s} | {'Qwen3-8B-RF':>18s}")
    print("  " + "-" * 75)
    for band in ["lowpass", "highpass", "bandpass"]:
        b1 = m_base["by_band"].get(band, {"total": 0, "order_ok": 0})
        b2 = m_ft["by_band"].get(band, {"total": 0, "order_ok": 0})
        print(f"  {band:30s} | {pct(b1['order_ok'], b1['total']):>18s} | {pct(b2['order_ok'], b2['total']):>18s}")

    # 数值误差对比
    print(f"\n  {'数值中位误差(%)':30s} | {'原始Qwen3-8B':>18s} | {'Qwen3-8B-RF':>18s}")
    print("  " + "-" * 75)
    all_fields = set(list(m_base["num_errs"].keys()) + list(m_ft["num_errs"].keys()))
    for f in sorted(all_fields):
        e1 = m_base["num_errs"].get(f, [])
        e2 = m_ft["num_errs"].get(f, [])
        med1 = f"{sorted(e1)[len(e1)//2]*100:.3f}" if e1 else "N/A"
        med2 = f"{sorted(e2)[len(e2)//2]*100:.3f}" if e2 else "N/A"
        print(f"  {f:30s} | {med1:>18s} | {med2:>18s}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="基线对比评估")
    parser.add_argument("--test_file", default=DEFAULT_TEST)
    parser.add_argument("--n", type=int, default=100, help="评估样本数 (0=全部)")
    args = parser.parse_args()

    # 读取测试集
    with open(args.test_file, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"测试集总样本: {len(samples)}")

    # 分类并采样（只取full类型用于公平对比）
    classified = []
    for s in samples:
        stype = classify_sample(s["messages"])
        classified.append((stype, s))

    by_type = defaultdict(list)
    for stype, s in classified:
        by_type[stype].append((stype, s))

    print("样本分布:", {k: len(v) for k, v in by_type.items()})

    # 采样
    if args.n and args.n < len(classified):
        total = len(classified)
        sampled = []
        for stype, items in by_type.items():
            n_t = max(1, int(args.n * len(items) / total))
            step = max(1, len(items) // n_t)
            sampled.extend(items[::step][:n_t])
        eval_samples = sampled
    else:
        eval_samples = classified

    print(f"评估样本数: {len(eval_samples)}\n")

    # === 1. 评估基线模型 ===
    print("=" * 60)
    print("  评估: 原始 Qwen3-8B (未微调)")
    print("=" * 60)
    model_base, tok_base = load_model(BASELINE_MODEL)
    m_base = evaluate_model(model_base, tok_base, eval_samples, "Qwen3-8B")
    print(f"  基线评估完成! 用时 {m_base['total_time']:.0f}s")

    # 释放显存
    del model_base, tok_base
    torch.cuda.empty_cache()

    # === 2. 评估微调模型 ===
    print("\n" + "=" * 60)
    print("  评估: Qwen3-8B-RF (微调后)")
    print("=" * 60)
    model_ft, tok_ft = load_model(FINETUNED_MODEL)
    m_ft = evaluate_model(model_ft, tok_ft, eval_samples, "Qwen3-8B-RF")
    print(f"  微调评估完成! 用时 {m_ft['total_time']:.0f}s")

    del model_ft, tok_ft
    torch.cuda.empty_cache()

    # === 3. 打印对比 ===
    print_comparison(m_base, m_ft)

    # === 4. 保存结果 ===
    out_path = Path("paper_materials/baseline_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换 defaultdict 为普通 dict
    def to_dict(m):
        m_copy = dict(m)
        m_copy["num_errs"] = {k: v for k, v in m["num_errs"].items()}
        m_copy["by_band"] = {k: dict(v) for k, v in m["by_band"].items()}
        del m_copy["raw_outputs"]
        return m_copy

    result = {
        "baseline": to_dict(m_base),
        "finetuned": to_dict(m_ft),
        "baseline_outputs": m_base["raw_outputs"],
        "finetuned_outputs": m_ft["raw_outputs"],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
