"""
评估 Qwen3-8B LoRA checkpoint 在滤波器设计任务上的效果。

用法:
    conda activate llmads
    cd G:\wenlong\llmrf
    python eval_checkpoint.py [--n 200] [--checkpoint <path>]
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_BASE   = r"G:\wenlong\models\Qwen3-8B"
DEFAULT_CKPT   = r"G:\wenlong\llmrf\LLaMA-Factory\saves\Qwen3-8B-Base\lora\train_q4_24g_safe\checkpoint-1200"
DEFAULT_TEST   = r"G:\wenlong\llmrf\LLaMA-Factory\data\filter_sft_zhmix\test.jsonl"
SYSTEM_PROMPT  = (
    "你是滤波器设计助手，支持低通(LPF)、高通(HPF)、带通(BPF)滤波器设计。"
    "当用户信息完整时，只输出一个JSON对象。"
    "低通/高通: 键必须为 filter_type, filter_band, ripple_db, fc, fs, R0, La_target, order。"
    "带通: 键必须为 filter_type, filter_band, ripple_db, f_center, bandwidth, fs_lower, fs_upper, R0, La_target, order。"
    "filter_band 取值: lowpass / highpass / bandpass。"
    "当用户缺参时，先追问缺失参数（仅追问缺失项）。绝不追问order；若用户未提供order，你需要自动补全order。"
    "单位要求: fc/fs/f_center/bandwidth 用Hz，R0用ohm，ripple_db/La_target用dB。"
)

# 数值字段（相对误差）
NUM_FIELDS_LPF_HPF = ["fc", "fs", "ripple_db", "R0", "La_target"]
NUM_FIELDS_BPF     = ["f_center", "bandwidth", "fs_lower", "fs_upper", "ripple_db", "R0", "La_target"]

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def rel_err(pred, gt):
    if gt == 0:
        return abs(pred - gt)
    return abs(pred - gt) / abs(gt)


def extract_json(text: str):
    """从模型输出中提取第一个 JSON 对象，兼容 <think> 标签。"""
    # 去掉 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 提取 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def build_messages(user_msg: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def load_model_tokenizer(base_path, ckpt_path, use_4bit=True):
    print(f"[加载] base model: {base_path}")
    print(f"[加载] LoRA adapter: {ckpt_path}")

    bnb_cfg = None
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    print("[加载] 完成\n")
    return model, tokenizer


def run_inference(model, tokenizer, messages, max_new_tokens=256):
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


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def evaluate(args):
    # 读取测试集
    with open(args.test_file, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    if args.n and args.n < len(samples):
        # 均匀采样，保持 LPF/HPF/BPF 分布
        step = len(samples) // args.n
        samples = samples[::step][: args.n]
    print(f"评估样本数: {len(samples)}\n")

    model, tokenizer = load_model_tokenizer(args.base, args.checkpoint, use_4bit=args.q4)

    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        msgs  = sample["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        gt_text  = next(m["content"] for m in msgs if m["role"] == "assistant")

        gt = None
        try:
            gt = json.loads(gt_text)
        except Exception:
            continue  # 跳过无效 GT

        pred_text = run_inference(model, tokenizer, build_messages(user_msg))
        pred = extract_json(pred_text)

        entry = {
            "idx": i,
            "user": user_msg,
            "gt": gt,
            "pred_raw": pred_text,
            "pred": pred,
            "json_ok": pred is not None,
        }

        if pred is not None:
            # 分类字段
            entry["filter_type_ok"] = pred.get("filter_type") == gt.get("filter_type")
            entry["filter_band_ok"] = pred.get("filter_band") == gt.get("filter_band")
            entry["order_ok"]       = pred.get("order")       == gt.get("order")

            # 数值字段
            band = gt.get("filter_band", "lowpass")
            num_fields = NUM_FIELDS_BPF if band == "bandpass" else NUM_FIELDS_LPF_HPF
            errs = {}
            for field in num_fields:
                gt_v   = gt.get(field)
                pred_v = pred.get(field)
                if gt_v is not None and pred_v is not None:
                    try:
                        e = rel_err(float(pred_v), float(gt_v))
                        errs[field] = e
                    except Exception:
                        pass
            entry["num_errs"] = errs
        else:
            entry["filter_type_ok"] = False
            entry["filter_band_ok"] = False
            entry["order_ok"]       = False
            entry["num_errs"]       = {}

        results.append(entry)

        # 进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"  [{i+1}/{len(samples)}] json_ok={entry['json_ok']} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # ─── 统计 ────────────────────────────────────────────────────────────────
    total = len(results)
    json_ok   = sum(r["json_ok"] for r in results)
    type_ok   = sum(r.get("filter_type_ok", False) for r in results)
    band_ok   = sum(r.get("filter_band_ok", False) for r in results)
    order_ok  = sum(r.get("order_ok", False) for r in results)

    print("\n" + "=" * 60)
    print(f"{'评估结果':^60}")
    print("=" * 60)
    print(f"  总样本:           {total}")
    print(f"  JSON解析成功率:   {json_ok/total*100:.1f}%  ({json_ok}/{total})")
    print(f"  filter_type 准确: {type_ok/total*100:.1f}%  ({type_ok}/{total})")
    print(f"  filter_band 准确: {band_ok/total*100:.1f}%  ({band_ok}/{total})")
    print(f"  order 准确:       {order_ok/total*100:.1f}%  ({order_ok}/{total})")

    # 各滤波类型分别统计
    by_band = defaultdict(list)
    for r in results:
        band = r["gt"].get("filter_band", "?")
        by_band[band].append(r)

    print("\n  ── 按 filter_band 分类 ──")
    for band, rs in sorted(by_band.items()):
        j  = sum(r["json_ok"] for r in rs)
        o  = sum(r.get("order_ok", False) for r in rs)
        print(f"  {band:12s}: n={len(rs):4d}  json={j/len(rs)*100:.1f}%  order={o/len(rs)*100:.1f}%")

    # 数值误差
    field_errs = defaultdict(list)
    for r in results:
        for field, e in r.get("num_errs", {}).items():
            field_errs[field].append(e)

    if field_errs:
        print("\n  ── 数值字段相对误差 (中位数 | 均值 | <1% 比例) ──")
        for field, errs in sorted(field_errs.items()):
            errs_s = sorted(errs)
            med    = errs_s[len(errs_s) // 2]
            mean   = sum(errs_s) / len(errs_s)
            lt1pct = sum(1 for e in errs_s if e < 0.01) / len(errs_s)
            print(f"  {field:15s}: median={med*100:6.2f}%  mean={mean*100:6.2f}%  <1%={lt1pct*100:.1f}%")

    # 写结果文件
    out_path = Path(args.checkpoint) / "eval_results.json"
    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": total,
        "json_ok_pct": json_ok / total,
        "filter_type_acc": type_ok / total,
        "filter_band_acc": band_ok / total,
        "order_acc": order_ok / total,
        "field_median_rel_err": {
            f: sorted(v)[len(v)//2] for f, v in field_errs.items()
        },
    }
    out_path.write_text(json.dumps(
        {"summary": summary, "details": results}, ensure_ascii=False, indent=2
    ), encoding="utf-8")
    print(f"\n  详细结果已保存: {out_path}")

    # ─── 错误样本展示 ─────────────────────────────────────────────────────────
    bad = [r for r in results if not r["json_ok"] or not r.get("order_ok")][:5]
    if bad:
        print("\n  ── 前5个出错样本 ──")
        for r in bad:
            print(f"  [idx={r['idx']}] user: {r['user'][:60]}...")
            print(f"    GT:   {json.dumps(r['gt'], ensure_ascii=False)}")
            print(f"    PRED: {r['pred_raw'][:120]}")
            print()

    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 LoRA checkpoint")
    parser.add_argument("--base",       default=DEFAULT_BASE,  help="基础模型路径")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,  help="LoRA checkpoint 路径")
    parser.add_argument("--test_file",  default=DEFAULT_TEST,  help="测试集 JSONL 路径")
    parser.add_argument("--n",          type=int, default=200, help="评估样本数 (0=全部)")
    parser.add_argument("--no_q4",      action="store_true",   help="不使用 4-bit 量化")
    args = parser.parse_args()
    args.q4 = not args.no_q4

    if args.n == 0:
        args.n = None

    evaluate(args)
