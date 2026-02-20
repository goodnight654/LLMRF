"""
LLM -> ADS filter design loop (Chebyshev only, print metrics).
Run with ADS Python environment.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from adsapi.llm_interface import LLMInterface
from adsapi.batch_filter_simulation import FilterSimulationPipeline


# ====== Config ======
DEFAULT_WORKSPACE_ROOT = r"G:\wenlong\ADS"
DEFAULT_R0 = 50.0
DEFAULT_FILTER_TYPE = "chebyshev"
OUTPUT_ROOT = Path(__file__).parent / "llm_ads_outputs"
# ====================


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _llm_prompt(context: str) -> str:
    return (
        "你是滤波器设计助手。请根据用户需求补全设计参数。\n"
        "输出严格 JSON，不要包含其他文字。\n"
        "\n"
        "规则：\n"
        "1) 如果参数不足，请输出 {\"questions\": [\"问题1\", \"问题2\"]}\n"
        "2) 如果参数已齐全，请输出以下字段：\n"
        "   {\"filter_type\": \"chebyshev\", \"ripple_db\": 0.1, \"fc\": 1.0e9, "
        "\"fs\": 2.0e9, \"R0\": 50, \"La\": 40, \"order\": 5}\n"
        "3) filter_type 固定为 chebyshev\n"
        "4) 单位：fc/fs 用 Hz，R0 欧姆，ripple_db/La 为 dB\n"
        "\n"
        f"用户输入：{context}"
    )


def _ensure_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    required = ["ripple_db", "fc", "fs", "R0", "La"]
    for key in required:
        if key not in spec:
            raise ValueError(f"缺少参数: {key}")
    spec["filter_type"] = DEFAULT_FILTER_TYPE
    if "R0" not in spec or spec["R0"] is None:
        spec["R0"] = DEFAULT_R0
    return spec


def main() -> int:
    llm = LLMInterface(model_type="local", model_name="qwen3:14b", verbose=True)

    print("输入滤波器需求，例如：低通，fc=1GHz，fs=2GHz，ripple=0.2dB")
    user_context = input("你的需求: ").strip()
    llm_raw = None
    while True:
        prompt = _llm_prompt(user_context)
        resp = llm.call(prompt, temperature=0.2, max_tokens=400)
        llm_raw = resp
        data = _extract_json(resp)
        if not data:
            print("LLM 输出无法解析，重试...")
            continue
        questions = data.get("questions") if isinstance(data, dict) else None
        if questions:
            for q in questions:
                ans = input(f"LLM 追问: {q} ").strip()
                user_context += f"\n{q} {ans}"
            continue
        try:
            spec = _ensure_spec(data)
            break
        except ValueError as exc:
            user_context += f"\n{exc}"
            continue

    # Run ADS simulation
    date_tag = time.strftime("%Y%m%d_%H%M%S")
    workspace_path = os.path.join(DEFAULT_WORKSPACE_ROOT, f"llm_ads_test_{date_tag}")
    output_dir = os.path.join(workspace_path, "results")

    pipeline = FilterSimulationPipeline(
        workspace_path=workspace_path,
        library_name=f"llm_ads_lib_{date_tag}",
        output_dir=output_dir,
    )

    spec["id"] = f"llm_{date_tag}"
    designs = pipeline.generator.design_batch([spec])
    results = pipeline.run_batch_simulation(designs)

    if not results:
        print("未生成有效仿真结果。")
        return 1

    metrics = results[0].get("metrics", {})
    output_payload = {
        "timestamp": date_tag,
        "user_input": user_context,
        "spec": spec,
        "metrics": metrics,
        "raw_result": results[0],
        "llm_raw": llm_raw,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    spec_path = OUTPUT_ROOT / f"spec_{date_tag}.json"
    result_path = OUTPUT_ROOT / f"result_{date_tag}.json"
    with spec_path.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, ensure_ascii=False)
    print("\n=== 仿真结果 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\n结果目录: {output_dir}")
    print(f"Spec JSON: {spec_path}")
    print(f"Result JSON: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
