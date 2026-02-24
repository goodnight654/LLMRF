"""
LLM -> ADS filter design closed-loop (v2)
- 使用微调后的 qwen3-8b-rf-agent (Ollama) 进行参数提取
- 支持 LPF / HPF / BPF (chebyshev)
- 闭环优化: 仿真 -> 分析误差 -> 修改 order/参数 -> 再次仿真
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from adsapi.batch_filter_simulation import FilterSimulationPipeline

DEFAULT_WORKSPACE_ROOT = r"G:\wenlong\ADS"
OUTPUT_ROOT = Path(__file__).parent / "llm_ads_outputs"

# Ollama 微调模型
OLLAMA_URL   = "http://localhost:11434/api/chat"
MODEL_NAME   = "qwen3-8b-rf-agent"

# 闭环优化参数
MAX_ITERATIONS = 5          # 最大优化迭代次数
S11_TARGET_DB  = -10.0      # S11 通带目标 (dB)
RIPPLE_TARGET_DB = 3.0      # 通带纹波目标 (dB)
S21_MIN_TARGET_DB = -3.0    # S21 通带最小值目标 (dB, BPF)
STOPBAND_TARGET_DB = -30.0  # 阻带 S21 抑制目标 (dB)

# 各 filter_band 所需字段
REQUIRED_FIELDS_LPF_HPF = ["filter_type", "filter_band", "ripple_db", "fc", "fs", "R0", "La_target", "order"]
REQUIRED_FIELDS_BPF = ["filter_type", "filter_band", "ripple_db", "f_center", "bandwidth",
                       "fs_lower", "fs_upper", "R0", "La_target", "order"]


# ─── Ollama Chat 接口 ─────────────────────────────────────────────────────────

def call_ollama(user_msg: str, system_msg: str = None,
                temperature: float = 0.2, max_tokens: int = 512) -> Optional[str]:
    """调用 Ollama chat API（使用微调模型内嵌的 system prompt）"""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
        else:
            print(f"[ERROR] Ollama 返回 {resp.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Ollama 调用失败: {e}")
        return None


# ─── JSON 提取 ────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取 JSON 对象"""
    if not text:
        return None
    # 去掉 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 直接 {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ─── 参数校验与转换 ───────────────────────────────────────────────────────────

def validate_and_normalize(spec: Dict[str, Any]) -> Dict[str, Any]:
    """校验 LLM 输出的 spec 并转换类型"""
    band = spec.get("filter_band", "lowpass")
    required = REQUIRED_FIELDS_BPF if band == "bandpass" else REQUIRED_FIELDS_LPF_HPF
    missing = [k for k in required if k not in spec or spec[k] in (None, "")]
    if missing:
        raise ValueError(f"缺少字段: {', '.join(missing)}")

    out = dict(spec)
    # filter_type 只支持 chebyshev / butterworth，不能是 band 名
    ft = str(out.get("filter_type", "chebyshev")).lower()
    if ft not in ("chebyshev", "butterworth"):
        ft = "chebyshev"
    out["filter_type"] = ft
    out["filter_band"] = str(out["filter_band"])
    out["ripple_db"] = float(out["ripple_db"])
    out["R0"] = float(out["R0"])
    out["order"] = int(out["order"])

    # La_target -> La (pipeline 字段名)
    out["La"] = float(out.pop("La_target", out.get("La", 40)))

    if band == "bandpass":
        for k in ("f_center", "bandwidth", "fs_lower", "fs_upper"):
            out[k] = float(out[k])

        # ── BPF bandwidth 自动修正 ──
        # LLM 有时将 bandwidth 误设为 fs_upper - fs_lower (阻带跨度)
        # 而非用户期望的通带 3dB 带宽。若 bandwidth > 合理上限，则修正
        fc = out["f_center"]
        bw = out["bandwidth"]
        fsl = out["fs_lower"]
        fsu = out["fs_upper"]
        stopband_span = fsu - fsl

        # 判据: band ratio > 15% 且 bandwidth ≈ stopband_span → 明显错误
        band_ratio = bw / fc if fc > 0 else 0
        if band_ratio > 0.15 and abs(bw - stopband_span) / stopband_span < 0.05:
            # 修正: 合理 BW ≈ 2/3 * 到最近阻带的距离
            reasonable_bw = min(fc - fsl, fsu - fc) * 1.0
            if reasonable_bw / fc > 0.15:
                reasonable_bw = fc * 0.08  # 极端时用 8%
            out["bandwidth"] = reasonable_bw
            print(f"  [修正] bandwidth {bw/1e6:.0f}MHz → {reasonable_bw/1e6:.0f}MHz "
                  f"(band ratio {band_ratio*100:.1f}% → {reasonable_bw/fc*100:.1f}%)")
    else:
        for k in ("fc", "fs"):
            out[k] = float(out[k])

    return out


# ─── 指标评估 ─────────────────────────────────────────────────────────────────

def evaluate_metrics(metrics: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """评估仿真指标是否达标，返回评估报告"""
    band = spec.get("filter_band", "lowpass")
    issues = []
    passed = True

    s11 = metrics.get("S11_max_dB", 0)
    ripple = metrics.get("passband_ripple_dB", 0)
    s21_stop = metrics.get("S21_stopband_max_dB", 0)

    if s11 > S11_TARGET_DB:
        issues.append(f"S11={s11:.1f}dB > {S11_TARGET_DB}dB (匹配差)")
        passed = False

    if ripple > RIPPLE_TARGET_DB:
        issues.append(f"纹波={ripple:.1f}dB > {RIPPLE_TARGET_DB}dB")
        passed = False

    if s21_stop is not None and s21_stop > STOPBAND_TARGET_DB:
        issues.append(f"阻带S21={s21_stop:.1f}dB > {STOPBAND_TARGET_DB}dB (抑制不够)")
        passed = False

    if band == "bandpass":
        s21_min = metrics.get("S21_passband_min_dB", -999)
        if s21_min < S21_MIN_TARGET_DB:
            issues.append(f"通带S21_min={s21_min:.1f}dB < {S21_MIN_TARGET_DB}dB (插损大)")
            passed = False

    return {"passed": passed, "issues": issues, "metrics_summary": {
        "S11_max_dB": s11,
        "passband_ripple_dB": ripple,
        "S21_stopband_max_dB": s21_stop,
        "S21_passband_min_dB": metrics.get("S21_passband_min_dB"),
    }}


# ─── 闭环优化 ─────────────────────────────────────────────────────────────────

def optimize_spec(spec: Dict[str, Any], eval_result: Dict[str, Any],
                  iteration: int = 1) -> Dict[str, Any]:
    """基于评估结果自动调整 spec
    
    策略优先级:
    1. 阻带抑制不够 → 提 order (上限 11)
    2. 纹波过大 → 降 order 或降 ripple_db
    3. S11 匹配差 → 降 ripple_db
    4. 插损大 → 降 order
    5. order 已到边界 → 调 ripple_db
    """
    new_spec = dict(spec)
    issues_text = "; ".join(eval_result["issues"])
    changed = False

    if "抑制不够" in issues_text or "阻带" in issues_text:
        if spec["order"] < 11:
            new_spec["order"] = min(spec["order"] + 2, 11)
            print(f"  [优化] order {spec['order']} → {new_spec['order']} (提升阻带抑制)")
            changed = True

    if not changed and "纹波" in issues_text:
        if spec["order"] > 3:
            new_spec["order"] = max(spec["order"] - 1, 3)
            print(f"  [优化] order {spec['order']} → {new_spec['order']} (降低纹波)")
            changed = True
        elif spec["ripple_db"] > 0.05:
            new_spec["ripple_db"] = max(spec["ripple_db"] * 0.5, 0.01)
            print(f"  [优化] ripple_db {spec['ripple_db']} → {new_spec['ripple_db']:.3f}")
            changed = True

    if not changed and "匹配差" in issues_text:
        if spec["ripple_db"] > 0.05:
            new_spec["ripple_db"] = max(spec["ripple_db"] * 0.5, 0.01)
            print(f"  [优化] ripple_db {spec['ripple_db']} → {new_spec['ripple_db']:.3f} (改善匹配)")
            changed = True

    if not changed and "插损" in issues_text:
        if spec["order"] > 3:
            new_spec["order"] = max(spec["order"] - 1, 3)
            print(f"  [优化] order {spec['order']} → {new_spec['order']} (降低插损)")
            changed = True

    # 兜底: 如果没有任何变化（order 到达边界），尝试调整 ripple
    if not changed:
        if iteration % 2 == 0 and spec["ripple_db"] > 0.05:
            new_spec["ripple_db"] = max(spec["ripple_db"] * 0.5, 0.01)
            print(f"  [优化] ripple_db {spec['ripple_db']} → {new_spec['ripple_db']:.3f} (兜底)")
        elif spec["order"] > 3:
            new_spec["order"] = spec["order"] - 2
            print(f"  [优化] order {spec['order']} → {new_spec['order']} (尝试低阶)")
        else:
            print(f"  [优化] 无可调参数，保持当前设计")

    return new_spec


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def print_banner():
    print("=" * 65)
    print("  LLM-ADS 闭环滤波器设计系统 v2")
    print(f"  模型: {MODEL_NAME} (Ollama)")
    print(f"  支持: LPF / HPF / BPF (Chebyshev)")
    print("=" * 65)


def main() -> int:
    print_banner()

    # ─── 1. 用户输入 ──────────────────────────────────────────────────────────
    print("\n示例:")
    print("  低通: 设计一个chebyshev低通滤波器，fc=1GHz，fs=2GHz，ripple=0.5dB，R0=50ohm，La_target=40dB")
    print("  带通: 设计一个中心频率2.4GHz，带宽200MHz的带通滤波器，纹波0.5dB，R0=50，La=40dB，下阻带2.1GHz，上阻带2.7GHz")
    print()
    user_input = input("你的需求: ").strip()
    if not user_input:
        print("未输入需求，退出。")
        return 1

    # ─── 2. LLM 提取参数 ──────────────────────────────────────────────────────
    print("\n[1/4] 调用 LLM 提取设计参数...")
    max_retry = 5
    spec = None

    for attempt in range(1, max_retry + 1):
        resp = call_ollama(user_input)
        if not resp:
            print(f"  第 {attempt} 次调用无响应，重试...")
            continue

        data = extract_json(resp)
        if not data:
            print(f"  第 {attempt} 次输出非 JSON: {resp[:80]}...")
            continue

        try:
            spec = validate_and_normalize(data)
            print(f"  参数提取成功 (第 {attempt} 次)")
            break
        except ValueError as e:
            print(f"  第 {attempt} 次参数不完整: {e}")
            # 追问缺失字段
            user_input += f"\n请补全: {e}"
            continue

    if spec is None:
        print("[FAIL] 无法从 LLM 提取有效参数，退出。")
        return 1

    print(f"\n  提取到的设计规格:")
    for k, v in spec.items():
        if k == "id":
            continue
        print(f"    {k}: {v}")

    # ─── 3. 闭环仿真优化 ──────────────────────────────────────────────────────
    print(f"\n[2/4] 开始闭环仿真优化 (最多 {MAX_ITERATIONS} 轮)...")
    date_tag = time.strftime("%Y%m%d_%H%M%S")
    iteration_log = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  ── 第 {iteration}/{MAX_ITERATIONS} 轮 ──")

        # 创建仿真 workspace
        ws_path = os.path.join(DEFAULT_WORKSPACE_ROOT, f"llm_ads_{date_tag}_iter{iteration}")
        out_dir = os.path.join(ws_path, "results")

        pipeline = FilterSimulationPipeline(
            workspace_path=ws_path,
            library_name=f"llm_lib_{date_tag}_i{iteration}",
            output_dir=out_dir,
        )

        current_spec = dict(spec)
        current_spec["id"] = f"llm_{date_tag}_i{iteration}"

        # 设计 & 仿真
        print(f"  设计参数: band={current_spec['filter_band']}, order={current_spec['order']}, "
              f"ripple={current_spec['ripple_db']}dB, R0={current_spec['R0']}Ω")

        try:
            designs = pipeline.generator.design_batch([current_spec])
        except Exception as e:
            print(f"  [WARN] 设计失败: {e}")
            # 可能 order 过大或参数不合理，降阶重试
            spec["order"] = max(spec["order"] - 1, 3)
            continue

        if not designs:
            print("  [WARN] 生成设计为空")
            continue

        results = pipeline.run_batch_simulation(designs)
        if not results:
            print("  [WARN] 仿真结果为空")
            continue

        metrics = results[0].get("metrics", {})

        # 评估
        eval_result = evaluate_metrics(metrics, spec)
        iter_record = {
            "iteration": iteration,
            "spec": dict(spec),
            "metrics": metrics,
            "eval": eval_result,
        }
        iteration_log.append(iter_record)

        # 打印关键指标
        print(f"  S11_max     = {metrics.get('S11_max_dB', 'N/A'):.2f} dB")
        print(f"  纹波        = {metrics.get('passband_ripple_dB', 'N/A'):.2f} dB")
        print(f"  阻带S21_max = {metrics.get('S21_stopband_max_dB', 'N/A'):.2f} dB")
        if spec.get("filter_band") == "bandpass":
            print(f"  通带S21_min = {metrics.get('S21_passband_min_dB', 'N/A'):.2f} dB")

        if eval_result["passed"]:
            print(f"\n  ✓ 第 {iteration} 轮全部指标达标！")
            break
        else:
            print(f"  ✗ 未达标: {'; '.join(eval_result['issues'])}")
            if iteration < MAX_ITERATIONS:
                spec = optimize_spec(spec, eval_result, iteration)
            else:
                print(f"\n  达到最大迭代次数 ({MAX_ITERATIONS})，使用当前最佳结果。")

    # ─── 4. 保存结果 ──────────────────────────────────────────────────────────
    print(f"\n[3/4] 保存结果...")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    final_result = {
        "timestamp": date_tag,
        "model": MODEL_NAME,
        "user_input": user_input,
        "final_spec": spec,
        "iterations": iteration_log,
        "total_iterations": len(iteration_log),
        "converged": iteration_log[-1]["eval"]["passed"] if iteration_log else False,
    }

    result_path = OUTPUT_ROOT / f"closed_loop_{date_tag}.json"
    json_str = json.dumps(final_result, indent=2, ensure_ascii=False, default=str)
    # 修复可能的 surrogate 字符
    json_str = json_str.encode("utf-8", errors="replace").decode("utf-8")
    result_path.write_text(json_str, encoding="utf-8")

    # ─── 5. 最终摘要 ──────────────────────────────────────────────────────────
    print(f"\n[4/4] 最终摘要")
    print("=" * 65)
    converged = final_result["converged"]
    print(f"  状态: {'✓ 收敛' if converged else '✗ 未完全收敛'}")
    print(f"  迭代次数: {final_result['total_iterations']}")
    print(f"  最终 order: {spec.get('order')}")

    if iteration_log:
        final_metrics = iteration_log[-1]["metrics"]
        print(f"\n  最终仿真指标:")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")

    print(f"\n  结果文件: {result_path}")
    print("=" * 65)
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
