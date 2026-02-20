r"""
滤波器数据集清洗脚本
python g:\wenlong\llmrf\dataset_cleanup.py 
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import csv


# 路径输出区域
INPUT_DIR = r"G:\wenlong\ADS\filter_dataset_results_20260212"
OUTPUT_JSON = r"G:\wenlong\ADS\filter_dataset_results_20260212\cleaned_dataset.json"
OUTPUT_CSV = r"G:\wenlong\ADS\filter_dataset_results_20260212\cleaned_summary.csv"
PREFER_FINAL_RESULTS = True

# 质量筛选阈值
STOPBAND_MARGIN_DB = 3.0
MAX_PASSBAND_RIPPLE_FACTOR = 3.0
MIN_PASSBAND_RIPPLE_DB = 0.5
MIN_S11_DB = -3.0
MAX_PASSBAND_LOSS_DB = 1.5



@dataclass
class QualityConfig:
    stopband_margin_db: float = 3.0
    max_passband_ripple_factor: float = 3.0
    min_passband_ripple_db: float = 0.5
    min_s11_db: float = -3.0
    max_passband_loss_db: float = 1.5


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v)


def _round_list(vals: Iterable[float], ndigits: int) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in vals)


def _spec_key(item: Dict[str, Any]) -> Tuple:
    """生成去重键（仅基于规格参数，不包含 L/C 元件值）"""
    design = item.get("design", {})
    params = design.get("params", {})
    ftype = design.get("filter_type", "chebyshev")
    return (
        ftype,
        int(design.get("N", 0)),
        round(float(params.get("ripple_db", 0.0)), 4),
        int(params.get("fc", 0)),
        int(params.get("fs", 0)),
        int(params.get("R0", 0)),
        int(params.get("La_target", 0)),
    )


def _score(item: Dict[str, Any]) -> float:
    metrics = item.get("metrics", {})
    stop = metrics.get("S21_stopband_max_dB", 0.0)
    ripple = metrics.get("passband_ripple_dB", 999.0)
    s11 = metrics.get("S11_max_dB", 0.0)
    return (-stop * 2.0) - ripple + (-s11 * 0.2)


def _passes_quality(item: Dict[str, Any], cfg: QualityConfig) -> bool:
    design = item.get("design", {})
    params = design.get("params", {})
    metrics = item.get("metrics", {})

    required_metrics = [
        "S11_max_dB",
        "S21_passband_min_dB",
        "S21_stopband_max_dB",
        "passband_ripple_dB",
    ]
    if any(k not in metrics for k in required_metrics):
        return False

    if not all(_is_number(metrics[k]) for k in required_metrics):
        return False

    fc = params.get("fc")
    fs = params.get("fs")
    if not _is_number(fc) or not _is_number(fs) or fs <= fc:
        return False

    la_target = params.get("La_target", 0)
    ripple_db = params.get("ripple_db", 0.0)

    max_ripple = max(cfg.min_passband_ripple_db, cfg.max_passband_ripple_factor * float(ripple_db))
    if metrics["passband_ripple_dB"] > max_ripple:
        return False

    if metrics["S21_passband_min_dB"] < -cfg.max_passband_loss_db:
        return False

    if metrics["S11_max_dB"] > cfg.min_s11_db:
        return False

    if metrics["S21_stopband_max_dB"] > -(float(la_target) - cfg.stopband_margin_db):
        return False

    return True


def _load_results(files: List[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                items.extend(data)
    return items


def _default_input_files(input_dir: Path) -> List[Path]:
    if PREFER_FINAL_RESULTS:
        final_file = input_dir / "final_results.json"
        if final_file.exists():
            return [final_file]

    patterns = ["results_batch_*.json", "final_results.json", "simulation_results.json"]
    files: List[Path] = []
    for p in patterns:
        files.extend(sorted(input_dir.glob(p)))
    return files


def main() -> int:
    input_dir = Path(INPUT_DIR)
    files = _default_input_files(input_dir)
    if not files:
        raise SystemExit(f"No result files found in {input_dir}")

    print("输入文件:")
    for f in files:
        print(f"  - {f.name}")

    cfg = QualityConfig(
        stopband_margin_db=STOPBAND_MARGIN_DB,
        max_passband_ripple_factor=MAX_PASSBAND_RIPPLE_FACTOR,
        min_passband_ripple_db=MIN_PASSBAND_RIPPLE_DB,
        min_s11_db=MIN_S11_DB,
        max_passband_loss_db=MAX_PASSBAND_LOSS_DB,
    )

    raw = _load_results(files)

    # Deduplicate by spec key; keep best score
    dedup: Dict[Tuple, Dict[str, Any]] = {}
    for item in raw:
        key = _spec_key(item)
        if key not in dedup:
            dedup[key] = item
        else:
            if _score(item) > _score(dedup[key]):
                dedup[key] = item

    filtered: List[Dict[str, Any]] = []
    for item in dedup.values():
        if _passes_quality(item, cfg):
            filtered.append(item)

    # Assign global IDs
    for idx, item in enumerate(filtered):
        ftype = item.get("design", {}).get("filter_type", "chebyshev")
        item["dataset_id"] = f"{ftype}_{idx:06d}"

    output_json = Path(OUTPUT_JSON)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(filtered, fh, indent=2, ensure_ascii=False)

    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset_id",
                "design_id",
                "filter_type",
                "order",
                "fc_hz",
                "fs_hz",
                "R0",
                "ripple_db",
                "La_target",
                "S11_max_dB",
                "S21_passband_min_dB",
                "S21_stopband_max_dB",
                "passband_ripple_dB",
            ],
        )
        writer.writeheader()
        for item in filtered:
            design = item.get("design", {})
            params = design.get("params", {})
            metrics = item.get("metrics", {})
            writer.writerow(
                {
                    "dataset_id": item.get("dataset_id"),
                    "design_id": item.get("design_id"),
                    "filter_type": design.get("filter_type", "chebyshev"),
                    "order": design.get("N"),
                    "fc_hz": params.get("fc"),
                    "fs_hz": params.get("fs"),
                    "R0": params.get("R0"),
                    "ripple_db": params.get("ripple_db"),
                    "La_target": params.get("La_target"),
                    "S11_max_dB": metrics.get("S11_max_dB"),
                    "S21_passband_min_dB": metrics.get("S21_passband_min_dB"),
                    "S21_stopband_max_dB": metrics.get("S21_stopband_max_dB"),
                    "passband_ripple_dB": metrics.get("passband_ripple_dB"),
                }
            )

    print(f"原始样本数: {len(raw)}")
    print(f"去重后样本数: {len(dedup)}")
    print(f"质量筛选后样本数: {len(filtered)}")
    if len(raw) > 0:
        print(f"去重保留率: {len(dedup) / len(raw) * 100:.2f}%")
        print(f"质量筛选保留率(相对去重后): {len(filtered) / len(dedup) * 100:.2f}%")
        print(f"质量筛选保留率(相对原始): {len(filtered) / len(raw) * 100:.2f}%")
    print(f"JSON 输出: {output_json}")
    print(f"CSV 输出: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
