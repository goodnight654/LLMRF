"""
增强版评估脚本：支持多轮对话、按样本类型统计、自动发现最佳checkpoint。

用法:
    conda activate llmads
    cd G:\wenlong\llmrf
    python eval_checkpoint_v2.py [--n 200] [--checkpoint <path>]

支持的样本类型:
  - full:              参数完整→直接输出JSON
  - followup_question: 缺参→追问（不产出JSON，评估追问质量）
  - followup_resolve:  缺参→追问→补参→输出JSON（多轮推理）
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_BASE   = r"G:\wenlong\models\Qwen3-8B"
DEFAULT_CKPT   = r"G:\wenlong\llmrf\LLaMA-Factory\saves\Qwen3-8B-Base\lora\train_cleaned_v2"
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

# 数值字段
NUM_FIELDS_LPF_HPF = ["fc", "fs", "ripple_db", "R0", "La_target"]
NUM_FIELDS_BPF     = ["f_center", "bandwidth", "fs_lower", "fs_upper", "ripple_db", "R0", "La_target"]

# ─── 样本类型判定 ──────────────────────────────────────────────────────────────

def classify_sample(msgs):
    """根据消息结构判断样本类型。"""
    # 统计 assistant 消息
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    user_msgs      = [m for m in msgs if m["role"] == "user"]

    if len(assistant_msgs) == 0:
        return "invalid"

    first_asst = assistant_msgs[0]["content"]

    # 尝试解析第一个 assistant 回复为 JSON
    try:
        json.loads(first_asst)
        return "full"  # 第一个回复就是 JSON
    except Exception:
        pass

    # 第一个回复不是 JSON → 追问
    if len(assistant_msgs) == 1:
        return "followup_question"  # 只有追问，没有后续回复

    # 有第二个 assistant 回复 → 多轮
    if len(assistant_msgs) >= 2:
        return "followup_resolve"

    return "unknown"


def get_final_gt(msgs):
    """获取最终的 GT（最后一个 assistant 的 JSON 输出）。"""
    for m in reversed(msgs):
        if m["role"] == "assistant":
            try:
                return json.loads(m["content"])
            except Exception:
                return None
    return None


def get_followup_gt_text(msgs):
    """获取追问回复的 GT 文本（第一个 assistant 回复）。"""
    for m in msgs:
        if m["role"] == "assistant":
            return m["content"]
    return None


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def rel_err(pred, gt):
    if gt == 0:
        return abs(pred - gt)
    return abs(pred - gt) / abs(gt)


def extract_json(text: str):
    """从模型输出中提取第一个 JSON 对象。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def is_followup_response(text: str) -> bool:
    """判断模型输出是否为追问而非 JSON。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 如果能解析为 JSON，就不是追问
    if extract_json(text) is not None:
        return False
    # 包含追问关键词
    followup_keywords = ["请补充", "请提供", "缺失", "missing", "provide", "Please provide",
                         "补充以下", "还需要", "缺少"]
    return any(k in text for k in followup_keywords)


# ─── 模型加载与推理 ───────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_dir: str) -> str:
    """在训练输出目录中找到最新的 checkpoint。"""
    ckpt_path = Path(ckpt_dir)

    # 如果路径本身就是一个 checkpoint（包含 adapter_model.*）
    if (ckpt_path / "adapter_model.safetensors").exists() or \
       (ckpt_path / "adapter_model.bin").exists():
        return str(ckpt_path)

    # 查找子目录中的 checkpoint
    checkpoints = []
    for d in ckpt_path.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except ValueError:
                pass

    if not checkpoints:
        raise FileNotFoundError(f"未找到任何 checkpoint: {ckpt_dir}")

    # 返回最新的 checkpoint
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best = checkpoints[0]
    print(f"[自动选择] 最新 checkpoint: {best[1].name} (step {best[0]})")
    return str(best[1])


def load_model_tokenizer(base_path, ckpt_path, use_4bit=True):
    print(f"[加载] 基础模型: {base_path}")
    print(f"[加载] LoRA 适配器: {ckpt_path}")

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


# ─── 评估逻辑 ─────────────────────────────────────────────────────────────────

def eval_json_output(pred, gt):
    """评估 JSON 输出的一致性。"""
    if pred is None:
        return {
            "json_ok": False, "filter_type_ok": False,
            "filter_band_ok": False, "order_ok": False, "num_errs": {},
            "missing_keys": [], "extra_keys": [],
        }

    result = {
        "json_ok": True,
        "filter_type_ok": pred.get("filter_type") == gt.get("filter_type"),
        "filter_band_ok": pred.get("filter_band") == gt.get("filter_band"),
        "order_ok": pred.get("order") == gt.get("order"),
    }

    # 检查键完整性
    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())
    result["missing_keys"] = list(gt_keys - pred_keys)
    result["extra_keys"] = list(pred_keys - gt_keys)

    # 数值误差
    band = gt.get("filter_band", "lowpass")
    num_fields = NUM_FIELDS_BPF if band == "bandpass" else NUM_FIELDS_LPF_HPF
    errs = {}
    for f in num_fields:
        gt_v = gt.get(f)
        pred_v = pred.get(f)
        if gt_v is not None and pred_v is not None:
            try:
                errs[f] = rel_err(float(pred_v), float(gt_v))
            except Exception:
                pass
    result["num_errs"] = errs

    return result


def eval_followup_question(pred_text, gt_text):
    """评估追问质量。"""
    pred_is_followup = is_followup_response(pred_text)
    gt_is_followup = True  # GT 是追问

    # 检查模型是否正确地追问了（而不是直接给出 JSON）
    pred_json = extract_json(pred_text)

    result = {
        "correctly_asked": pred_is_followup,
        "wrongly_gave_json": pred_json is not None,
    }

    # 关键词匹配分析（检查是否追问了正确的参数）
    missing_params = []
    for param in ["La_target", "ripple_db", "filter_type", "fc", "fs",
                   "f_center", "bandwidth", "fs_lower", "fs_upper", "R0"]:
        if param.lower() in gt_text.lower():
            missing_params.append(param)

    pred_mentioned = []
    for param in missing_params:
        if param.lower() in pred_text.lower():
            pred_mentioned.append(param)

    result["gt_missing_params"] = missing_params
    result["pred_mentioned_params"] = pred_mentioned
    result["param_recall"] = len(pred_mentioned) / max(len(missing_params), 1)

    return result


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def evaluate(args):
    # 读取测试集
    with open(args.test_file, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"测试集总样本: {len(samples)}")

    # 分类
    by_type = defaultdict(list)
    for i, s in enumerate(samples):
        stype = classify_sample(s["messages"])
        by_type[stype].append((i, s))

    print("样本分布:")
    for t, ss in sorted(by_type.items()):
        print(f"  {t:20s}: {len(ss)}")
    print()

    # 采样
    if args.n and args.n < len(samples):
        # 按比例从各类型采样
        total = sum(len(v) for v in by_type.values())
        sampled = []
        for t, ss in by_type.items():
            n_t = max(1, int(args.n * len(ss) / total))
            step = max(1, len(ss) // n_t)
            sampled.extend([(t, idx, s) for idx, s in ss[::step][:n_t]])
        samples_eval = sampled
    else:
        samples_eval = [(classify_sample(s["messages"]), i, s) for i, s in enumerate(samples)]

    print(f"评估样本数: {len(samples_eval)}\n")

    # 找到最佳 checkpoint
    ckpt_path = find_best_checkpoint(args.checkpoint)
    model, tokenizer = load_model_tokenizer(args.base, ckpt_path, use_4bit=args.q4)

    # 评估
    results = {"full": [], "followup_question": [], "followup_resolve": []}
    t0 = time.time()

    for eval_idx, (stype, orig_idx, sample) in enumerate(samples_eval):
        msgs = sample["messages"]

        if stype == "full":
            # 单轮：user→assistant(JSON)
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            gt = get_final_gt(msgs)
            if gt is None:
                continue

            pred_text = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            pred = extract_json(pred_text)
            entry = {
                "idx": orig_idx, "type": "full",
                "user": user_msg, "gt": gt,
                "pred_raw": pred_text, "pred": pred,
                **eval_json_output(pred, gt),
            }
            results["full"].append(entry)

        elif stype == "followup_question":
            # 追问：user(缺参)→assistant(追问)
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            gt_text = get_followup_gt_text(msgs)
            if gt_text is None:
                continue

            pred_text = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            fq_result = eval_followup_question(pred_text, gt_text)
            entry = {
                "idx": orig_idx, "type": "followup_question",
                "user": user_msg, "gt_text": gt_text,
                "pred_raw": pred_text,
                **fq_result,
            }
            results["followup_question"].append(entry)

        elif stype == "followup_resolve":
            # 多轮：user(缺参)→assistant(追问)→user(补参)→assistant(JSON)
            user_msgs = [m for m in msgs if m["role"] == "user"]
            asst_msgs = [m for m in msgs if m["role"] == "assistant"]
            gt = get_final_gt(msgs)
            if gt is None or len(user_msgs) < 2 or len(asst_msgs) < 2:
                continue

            # 第一轮：追问
            first_user = user_msgs[0]["content"]
            gt_followup = asst_msgs[0]["content"]
            pred_followup = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user},
            ])
            fq_result = eval_followup_question(pred_followup, gt_followup)

            # 第二轮：补参后给出 JSON（使用 GT 追问以确保公平）
            second_user = user_msgs[1]["content"]
            pred_final = run_inference(model, tokenizer, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": gt_followup},  # 使用 GT 追问
                {"role": "user", "content": second_user},
            ])
            pred_json = extract_json(pred_final)
            json_result = eval_json_output(pred_json, gt)

            entry = {
                "idx": orig_idx, "type": "followup_resolve",
                "user_1": first_user, "user_2": second_user,
                "gt": gt, "gt_followup": gt_followup,
                "pred_followup": pred_followup,
                "pred_final_raw": pred_final,
                "pred_final": pred_json,
                "followup_eval": fq_result,
                **json_result,
            }
            results["followup_resolve"].append(entry)

        # 进度
        if (eval_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (eval_idx + 1) * (len(samples_eval) - eval_idx - 1)
            print(f"  [{eval_idx+1}/{len(samples_eval)}] "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    total_time = time.time() - t0

    # ─── 统计 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'评估结果 (增强版)':^70}")
    print("=" * 70)
    print(f"  评估总时间: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Checkpoint:  {ckpt_path}")

    # === 1. Full 类型 ===
    full = results["full"]
    if full:
        n = len(full)
        json_ok   = sum(r["json_ok"] for r in full)
        type_ok   = sum(r["filter_type_ok"] for r in full)
        band_ok   = sum(r["filter_band_ok"] for r in full)
        order_ok  = sum(r["order_ok"] for r in full)

        print(f"\n  ══ Full (参数完整→直接JSON) ══  [{n} 样本]")
        print(f"    JSON解析成功率:   {json_ok/n*100:.1f}%  ({json_ok}/{n})")
        print(f"    filter_type 准确: {type_ok/n*100:.1f}%")
        print(f"    filter_band 准确: {band_ok/n*100:.1f}%")
        print(f"    order 准确:       {order_ok/n*100:.1f}%")

        # 按 band 分类
        by_band = defaultdict(list)
        for r in full:
            band = r["gt"].get("filter_band", "?")
            by_band[band].append(r)

        print(f"\n    按 filter_band      | n    | json%  | order% | keys完整%")
        print(f"    " + "-" * 60)
        for band, rs in sorted(by_band.items()):
            j  = sum(r["json_ok"] for r in rs)
            o  = sum(r["order_ok"] for r in rs)
            mk = sum(1 for r in rs if not r.get("missing_keys"))
            print(f"    {band:20s} | {len(rs):4d} | {j/len(rs)*100:5.1f}% | "
                  f"{o/len(rs)*100:5.1f}% | {mk/len(rs)*100:5.1f}%")

        # 数值误差
        field_errs = defaultdict(list)
        for r in full:
            for f, e in r.get("num_errs", {}).items():
                field_errs[f].append(e)

        if field_errs:
            print(f"\n    数值字段       | median%  | mean%   | <1%比例 | <0.1%比例")
            print(f"    " + "-" * 60)
            for f, errs in sorted(field_errs.items()):
                errs_s = sorted(errs)
                med    = errs_s[len(errs_s) // 2]
                mean   = sum(errs_s) / len(errs_s)
                lt1pct = sum(1 for e in errs_s if e < 0.01) / len(errs_s)
                lt01   = sum(1 for e in errs_s if e < 0.001) / len(errs_s)
                print(f"    {f:15s} | {med*100:7.3f} | {mean*100:7.3f} | "
                      f"{lt1pct*100:6.1f}% | {lt01*100:6.1f}%")

    # === 2. Followup Question 类型 ===
    fq = results["followup_question"]
    if fq:
        n = len(fq)
        correctly_asked = sum(r["correctly_asked"] for r in fq)
        wrongly_json    = sum(r["wrongly_gave_json"] for r in fq)
        avg_recall      = sum(r["param_recall"] for r in fq) / n

        print(f"\n  ══ Followup Question (追问) ══  [{n} 样本]")
        print(f"    正确追问率:       {correctly_asked/n*100:.1f}%  ({correctly_asked}/{n})")
        print(f"    错误输出JSON率:   {wrongly_json/n*100:.1f}%")
        print(f"    追问参数召回率:   {avg_recall*100:.1f}%")

    # === 3. Followup Resolve 类型 ===
    fr = results["followup_resolve"]
    if fr:
        n = len(fr)
        json_ok_fr = sum(r["json_ok"] for r in fr)
        order_ok_fr = sum(r["order_ok"] for r in fr)
        fu_correct  = sum(r["followup_eval"]["correctly_asked"] for r in fr)

        print(f"\n  ══ Followup Resolve (多轮补参) ══  [{n} 样本]")
        print(f"    追问正确率 (round1): {fu_correct/n*100:.1f}%")
        print(f"    JSON解析成功率 (round2): {json_ok_fr/n*100:.1f}%")
        print(f"    order 准确率 (round2):   {order_ok_fr/n*100:.1f}%")

        # 数值误差
        field_errs_fr = defaultdict(list)
        for r in fr:
            for f, e in r.get("num_errs", {}).items():
                field_errs_fr[f].append(e)

        if field_errs_fr:
            print(f"\n    数值字段       | median%  | mean%   | <1%比例")
            print(f"    " + "-" * 48)
            for f, errs in sorted(field_errs_fr.items()):
                errs_s = sorted(errs)
                med    = errs_s[len(errs_s) // 2]
                mean   = sum(errs_s) / len(errs_s)
                lt1pct = sum(1 for e in errs_s if e < 0.01) / len(errs_s)
                print(f"    {f:15s} | {med*100:7.3f} | {mean*100:7.3f} | {lt1pct*100:6.1f}%")

    # === 综合指标 ===
    all_json = results["full"] + results["followup_resolve"]
    if all_json:
        total_json = len(all_json)
        total_ok = sum(r["json_ok"] for r in all_json)
        total_order = sum(r["order_ok"] for r in all_json)

        print(f"\n  ══ 综合指标 (所有需JSON输出的样本) ══")
        print(f"    总数: {total_json}")
        print(f"    JSON解析总成功率: {total_ok/total_json*100:.1f}%")
        print(f"    order 总准确率:   {total_order/total_json*100:.1f}%")

        # 综合数值误差
        all_errs = defaultdict(list)
        for r in all_json:
            for f, e in r.get("num_errs", {}).items():
                all_errs[f].append(e)
        if all_errs:
            avg_median = sum(sorted(v)[len(v)//2] for v in all_errs.values()) / len(all_errs)
            print(f"    平均中位数相对误差: {avg_median*100:.3f}%")

    # 写结果文件
    out_path = Path(ckpt_path) / "eval_results_v2.json"
    summary = {
        "checkpoint": ckpt_path,
        "n_full": len(results["full"]),
        "n_followup_q": len(results["followup_question"]),
        "n_followup_r": len(results["followup_resolve"]),
        "total_time_s": total_time,
    }
    if full:
        summary["full_json_ok_pct"] = sum(r["json_ok"] for r in full) / len(full)
        summary["full_order_acc"] = sum(r["order_ok"] for r in full) / len(full)
    if fq:
        summary["fq_correctly_asked_pct"] = sum(r["correctly_asked"] for r in fq) / len(fq)
    if fr:
        summary["fr_json_ok_pct"] = sum(r["json_ok"] for r in fr) / len(fr)
        summary["fr_order_acc"] = sum(r["order_ok"] for r in fr) / len(fr)

    out_path.write_text(json.dumps(
        {"summary": summary, "details": {
            "full": results["full"],
            "followup_question": results["followup_question"],
            "followup_resolve": results["followup_resolve"],
        }}, ensure_ascii=False, indent=2, default=str
    ), encoding="utf-8")
    print(f"\n  结果已保存: {out_path}")

    # 错误样本展示
    bad_full = [r for r in full if not r["json_ok"] or not r["order_ok"]][:3]
    bad_fq   = [r for r in fq if not r["correctly_asked"]][:2]

    if bad_full:
        print(f"\n  ── Full 类型错误样本 ──")
        for r in bad_full:
            print(f"  [idx={r['idx']}] user: {r['user'][:80]}...")
            print(f"    GT:   order={r['gt'].get('order')}  band={r['gt'].get('filter_band')}")
            print(f"    PRED: {r['pred_raw'][:120]}")
            print()

    if bad_fq:
        print(f"\n  ── Followup Question 错误样本 ──")
        for r in bad_fq:
            print(f"  [idx={r['idx']}] user: {r['user'][:80]}...")
            print(f"    GT:   {r['gt_text'][:100]}")
            print(f"    PRED: {r['pred_raw'][:120]}")
            print()

    print("=" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版 LoRA checkpoint 评估")
    parser.add_argument("--base",       default=DEFAULT_BASE,  help="基础模型路径")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,  help="LoRA checkpoint 路径或训练输出目录")
    parser.add_argument("--test_file",  default=DEFAULT_TEST,  help="测试集 JSONL 路径")
    parser.add_argument("--n",          type=int, default=200, help="评估样本数 (0=全部)")
    parser.add_argument("--no_q4",      action="store_true",   help="不使用 4-bit 量化")
    args = parser.parse_args()
    args.q4 = not args.no_q4

    if args.n == 0:
        args.n = None

    evaluate(args)
