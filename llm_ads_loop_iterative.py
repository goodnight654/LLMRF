"""
LLM -> ADS filter design loop (Iterative Version)
- 基于 adsapi.llm_interface.LLMInterface
- 支持缺参追问与自动补全
- 核心创新：实现“结果评估与反馈重试”的闭环反思机制 (Reflection)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adsapi.llm_interface import LLMInterface
from adsapi.batch_filter_simulation import FilterSimulationPipeline


# ======================== 配置区 ========================
DEFAULT_WORKSPACE_ROOT = r"G:\wenlong\ADS"
DEFAULT_R0 = 50.0
DEFAULT_FILTER_TYPE = "chebyshev"
DEFAULT_ORDER = 5
OUTPUT_ROOT = Path(__file__).parent / "llm_ads_outputs"
MODEL_NAME = "qwen3:8b" # 准备使用微调后的 8B 模型
MAX_ITERATIONS = 5      # 最大迭代优化次数
# ========================================================

REQUIRED_FIELDS = ["ripple_db", "fc", "fs", "R0", "La"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _missing_fields(spec: Dict[str, Any]) -> List[str]:
    return [k for k in REQUIRED_FIELDS if k not in spec or spec[k] in (None, "")]


def _llm_prompt_initial(context: str) -> str:
    return (
        "你是专业的射频滤波器设计助手。请根据用户需求提取或补全设计参数。\n"
        "输出必须是可解析JSON，不要输出额外文字。\n\n"
        "规则：\n"
        "1) 参数不足时，输出: {\"questions\": [\"问题1\", \"问题2\"]}\n"
        "2) 参数齐全时，输出字段：\n"
        "   {\"filter_type\": \"chebyshev\", \"ripple_db\": 0.1, \"fc\": 1.0e9, \"fs\": 2.0e9, \"R0\": 50, \"La\": 40, \"order\": 5}\n"
        "3) 如果用户没给 order，不要追问 order，直接自动补全一个合理 order。\n"
        "4) filter_type 固定为 chebyshev。\n"
        "5) 单位: fc/fs 用Hz，R0欧姆，ripple_db/La 用dB。\n\n"
        f"用户输入:\n{context}"
    )

def _llm_prompt_reflection(original_spec: Dict[str, Any], metrics: Dict[str, Any], target_la: float) -> str:
    """
    反思提示词：将上一次的参数和仿真结果喂给 LLM，让其调整参数。
    """
    return (
        "你是专业的射频滤波器设计专家。上一轮的滤波器参数仿真结果未达到目标。\n"
        f"【上一轮参数】: {json.dumps(original_spec, ensure_ascii=False)}\n"
        f"【仿真结果】: {json.dumps(metrics, ensure_ascii=False)}\n"
        f"【目标阻带衰减 (La)】: {target_la} dB\n\n"
        "请分析仿真结果。如果阻带衰减 (La) 未达标，通常需要增加阶数 (order) 或调整截止频率 (fc)。\n"
        "请输出调整后的全新参数 JSON。输出必须是可解析JSON，不要输出额外文字。\n"
        "必须包含字段：filter_type, ripple_db, fc, fs, R0, La, order。\n"
    )


def _ensure_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    miss = _missing_fields(spec)
    if miss:
        raise ValueError("缺少参数: " + ", ".join(miss))

    out = dict(spec)
    out["filter_type"] = DEFAULT_FILTER_TYPE

    if out.get("R0") in (None, ""):
        out["R0"] = DEFAULT_R0

    if out.get("order") in (None, ""):
        out["order"] = DEFAULT_ORDER

    out["ripple_db"] = float(out["ripple_db"])
    out["fc"] = float(out["fc"])
    out["fs"] = float(out["fs"])
    out["R0"] = float(out["R0"])
    out["La"] = float(out["La"])
    out["order"] = int(out["order"])

    return out

def evaluate_metrics(metrics: Dict[str, Any], target_spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    评估仿真结果是否达标。
    这里以阻带衰减 (La) 为例进行评估。
    """
    # 假设 metrics 中有类似 'stopband_attenuation' 或我们可以从 S21 提取
    # 注意：这里的键名需要根据你实际 ADS 返回的 metrics 字典结构来调整
    
    # 这是一个简化的评估逻辑，你需要根据实际的 metrics 结构修改
    is_pass = True
    reason = "仿真结果达标。"
    
    # 示例：检查阻带衰减是否达到目标 La
    # 假设 metrics 里有一个键叫 'min_stopband_attenuation_db'
    actual_la = metrics.get("min_stopband_attenuation_db") 
    target_la = target_spec.get("La", 0)
    
    if actual_la is not None:
        # 注意：衰减通常是正数，或者 S21 是负数。这里假设 actual_la 是正数表示衰减了多少 dB
        if actual_la < target_la:
            is_pass = False
            reason = f"阻带衰减不足: 实际 {actual_la:.2f} dB < 目标 {target_la} dB"
    else:
        # 如果没有直接的指标，可能需要更复杂的判断，这里暂时放行或标记为需要人工确认
        print("警告: 未在 metrics 中找到阻带衰减指标，默认放行。请检查 ADS 提取逻辑。")
        
    return is_pass, reason


def main() -> int:
    llm = LLMInterface(model_type="local", model_name=MODEL_NAME, verbose=True)

    print("输入需求示例：低通，fc=1GHz，fs=2GHz，ripple=0.2dB，La=40dB，R0=50")
    user_context = input("你的需求: ").strip()

    # ================= 阶段 1：初始参数解析 =================
    llm_raw = None
    retry_count = 0
    max_retry = 8
    spec = {}

    while True:
        retry_count += 1
        if retry_count > max_retry:
            print("超过最大重试次数，退出。")
            return 1

        resp = llm.call(_llm_prompt_initial(user_context), temperature=0.2, max_tokens=500)
        llm_raw = resp
        data = _extract_json(resp or "")

        if not data:
            print("LLM 输出无法解析 JSON，重试中...")
            continue

        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            questions = [str(q).strip() for q in data["questions"] if str(q).strip()]
            questions = [q for q in questions if "order" not in q.lower() and "阶数" not in q]
            if questions:
                for q in questions:
                    ans = input(f"LLM 追问: {q} ").strip()
                    user_context += f"\n{q} {ans}"
                continue

        try:
            spec = _ensure_spec(data)
            break
        except ValueError as exc:
            print(f"参数仍不完整: {exc}")
            user_context += f"\n{exc}"
            continue
        except Exception as exc:
            print(f"参数格式错误: {exc}")
            continue

    # ================= 阶段 2：迭代仿真与反思 =================
    date_tag = time.strftime("%Y%m%d_%H%M%S")
    workspace_path = os.path.join(DEFAULT_WORKSPACE_ROOT, f"llm_ads_test_{date_tag}")
    output_dir = os.path.join(workspace_path, "results")

    pipeline = FilterSimulationPipeline(
        workspace_path=workspace_path,
        library_name=f"llm_ads_lib_{date_tag}",
        output_dir=output_dir,
    )

    iteration = 1
    history = [] # 记录迭代历史

    while iteration <= MAX_ITERATIONS:
        print(f"\n>>> 开始第 {iteration} 轮仿真迭代...")
        spec["id"] = f"llm_{date_tag}_iter{iteration}"
        
        # 1. 运行仿真
        designs = pipeline.generator.design_batch([spec])
        results = pipeline.run_batch_simulation(designs)

        if not results:
            print("未生成有效仿真结果，迭代终止。")
            return 1

        metrics = results[0].get("metrics", {})
        
        print("\n=== 当前仿真结果 ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # 2. 评估结果
        is_pass, reason = evaluate_metrics(metrics, spec)
        
        # 记录历史
        history.append({
            "iteration": iteration,
            "spec": spec.copy(),
            "metrics": metrics,
            "evaluation": reason
        })

        if is_pass:
            print(f"\n✅ 仿真达标！({reason})")
            break
        else:
            print(f"\n❌ 仿真未达标: {reason}")
            if iteration == MAX_ITERATIONS:
                print("达到最大迭代次数，优化失败。")
                break
                
            # 3. LLM 反思与参数修正
            print("正在请求 LLM 进行反思和参数修正...")
            target_la = spec.get("La", 0)
            reflection_prompt = _llm_prompt_reflection(spec, metrics, target_la)
            
            resp = llm.call(reflection_prompt, temperature=0.3, max_tokens=500)
            new_data = _extract_json(resp or "")
            
            if new_data:
                try:
                    spec = _ensure_spec(new_data)
                    print(f"LLM 提出了新的参数: {spec}")
                except Exception as e:
                    print(f"LLM 返回的新参数格式错误: {e}，将使用原参数重试或手动干预。")
                    # 这里可以加入更复杂的错误恢复逻辑
            else:
                print("LLM 未能返回有效的 JSON 修正参数。")

        iteration += 1

    # ================= 阶段 3：保存最终结果 =================
    output_payload = {
        "timestamp": date_tag,
        "user_input": user_context,
        "final_spec": spec,
        "final_metrics": metrics,
        "iteration_history": history,
        "llm_raw_initial": llm_raw,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    spec_path = OUTPUT_ROOT / f"spec_{date_tag}.json"
    result_path = OUTPUT_ROOT / f"result_{date_tag}.json"
    
    with spec_path.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, ensure_ascii=False)

    print(f"\n结果目录: {output_dir}")
    print(f"Spec JSON: {spec_path}")
    print(f"Result JSON: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
