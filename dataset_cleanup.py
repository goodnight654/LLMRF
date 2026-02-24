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
INPUT_DIR = r"G:\wenlong\ADS\filter_dataset_results_20260224"
OUTPUT_JSON = r"G:\wenlong\ADS\filter_dataset_results_20260224\cleaned_dataset.json"
OUTPUT_CSV = r"G:\wenlong\ADS\filter_dataset_results_20260224\cleaned_summary.csv"
PREFER_FINAL_RESULTS = True

# 质量筛选阈值 (LPF / HPF)
STOPBAND_MARGIN_DB = 5.0           # 较宽松的相对裕量
STOPBAND_ABS_MIN_DB = -8.0         # 绝对最低阻带衰减 (dB);任何设计至少应有 8dB
MAX_PASSBAND_RIPPLE_FACTOR = 5.0   # 允许纹波为目标的 5 倍
MIN_PASSBAND_RIPPLE_DB = 2.0       # 纹波上限至少 2 dB
MIN_S11_DB = -2.0                  # 回波损耗 < -2 dB
MAX_PASSBAND_LOSS_DB = 3.0         # 通带损耗 < 3 dB

# BPF 专用阈值 (带通滤波器固有损耗更高，适当放宽)
BPF_MAX_PASSBAND_LOSS_DB = 8.0     # BPF 通带损耗允许 8 dB
BPF_MIN_S11_DB = -0.5              # BPF S11 < -0.5 dB (排除完全反射)
BPF_MAX_PASSBAND_RIPPLE_FACTOR = 8.0
BPF_MIN_PASSBAND_RIPPLE_DB = 3.0
BPF_STOPBAND_MARGIN_DB = 5.0
BPF_STOPBAND_ABS_MIN_DB = -3.0     # BPF 阻带至少 3 dB 衰减



@dataclass
class QualityConfig:
    """LPF/HPF 质量筛选配置"""
    stopband_margin_db: float = 5.0
    stopband_abs_min_db: float = -8.0
    max_passband_ripple_factor: float = 5.0
    min_passband_ripple_db: float = 2.0
    min_s11_db: float = -2.0
    max_passband_loss_db: float = 3.0


@dataclass
class BPFQualityConfig:
    """BPF 质量筛选配置（带通固有损耗更高，适当放宽）"""
    stopband_margin_db: float = 5.0
    stopband_abs_min_db: float = -3.0
    max_passband_ripple_factor: float = 8.0
    min_passband_ripple_db: float = 3.0
    min_s11_db: float = -0.5
    max_passband_loss_db: float = 8.0


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v)


def _round_list(vals: Iterable[float], ndigits: int) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in vals)


def _get_filter_band(item: Dict[str, Any]) -> str:
    """从 design 或 metrics 获取 filter_band"""
    band = item.get("design", {}).get("filter_band")
    if not band:
        band = item.get("metrics", {}).get("filter_band", "lowpass")
    return band


def _spec_key(item: Dict[str, Any]) -> Tuple:
    """生成去重键（仅基于规格参数，不包含 L/C 元件值），支持 LPF/HPF/BPF"""
    design = item.get("design", {})
    params = design.get("params", {})
    ftype = design.get("filter_type", "chebyshev")
    band = _get_filter_band(item)

    base = (
        ftype,
        band,
        int(design.get("N", 0)),
        round(float(params.get("ripple_db", 0.0)), 4),
        int(params.get("R0", 0)),
        int(params.get("La_target", 0)),
    )

    if band == "bandpass":
        return base + (
            int(params.get("f_center", 0)),
            int(params.get("bandwidth", 0)),
        )
    else:
        return base + (
            int(params.get("fc", 0)),
            int(params.get("fs", 0)),
        )


def _score(item: Dict[str, Any]) -> float:
    metrics = item.get("metrics", {})
    stop = metrics.get("S21_stopband_max_dB", 0.0)
    ripple = metrics.get("passband_ripple_dB", 999.0)
    s11 = metrics.get("S11_max_dB", 0.0)
    return (-stop * 2.0) - ripple + (-s11 * 0.2)


def _passes_quality(item: Dict[str, Any], cfg: QualityConfig,
                    bpf_cfg: Optional[BPFQualityConfig] = None) -> bool:
    """质量筛选，支持 LPF/HPF/BPF 不同阈值"""
    design = item.get("design", {})
    params = design.get("params", {})
    metrics = item.get("metrics", {})
    band = _get_filter_band(item)

    required_metrics = [
        "S11_max_dB",
        "S21_passband_min_dB",
        "passband_ripple_dB",
    ]
    # S21_stopband_max_dB 可选（有些 BPF 可能缺失）
    if any(k not in metrics for k in required_metrics):
        return False
    if not all(_is_number(metrics[k]) for k in required_metrics):
        return False

    # 频率合理性检查
    if band == "bandpass":
        f_center = params.get("f_center")
        bandwidth = params.get("bandwidth")
        if not _is_number(f_center) or not _is_number(bandwidth) or bandwidth <= 0:
            return False
    elif band == "highpass":
        fc = params.get("fc")
        fs = params.get("fs")
        if not _is_number(fc) or not _is_number(fs) or fs >= fc:
            return False  # HPF: fs < fc
    else:  # lowpass
        fc = params.get("fc")
        fs = params.get("fs")
        if not _is_number(fc) or not _is_number(fs) or fs <= fc:
            return False  # LPF: fs > fc

    la_target = float(params.get("La_target", 0))
    ripple_db = float(params.get("ripple_db", 0.0))

    # 根据频带选择阈值
    if band == "bandpass" and bpf_cfg is not None:
        max_ripple = max(bpf_cfg.min_passband_ripple_db,
                         bpf_cfg.max_passband_ripple_factor * ripple_db)
        max_loss = bpf_cfg.max_passband_loss_db
        min_s11 = bpf_cfg.min_s11_db
        stop_margin = bpf_cfg.stopband_margin_db
    else:
        max_ripple = max(cfg.min_passband_ripple_db,
                         cfg.max_passband_ripple_factor * ripple_db)
        max_loss = cfg.max_passband_loss_db
        min_s11 = cfg.min_s11_db
        stop_margin = cfg.stopband_margin_db

    # 通带纹波检查
    if metrics["passband_ripple_dB"] > max_ripple:
        return False

    # 通带损耗检查
    if metrics["S21_passband_min_dB"] < -max_loss:
        return False

    # 回波损耗 (S11) 检查
    if metrics["S11_max_dB"] > min_s11:
        return False

    # 阻带衰减检查: 仅检查绝对下限（确保有基本滤波效果）
    # 不检查相对La_target, 因为理论目标值通常远高于实际可实现值
    stop_val = metrics.get("S21_stopband_max_dB")
    if band == "bandpass" and bpf_cfg is not None:
        abs_min = bpf_cfg.stopband_abs_min_db
    else:
        abs_min = cfg.stopband_abs_min_db

    if _is_number(stop_val):
        # 绝对下限: 阻带至少应有一定衰减
        if stop_val > abs_min:
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
    bpf_cfg = BPFQualityConfig(
        stopband_margin_db=BPF_STOPBAND_MARGIN_DB,
        stopband_abs_min_db=BPF_STOPBAND_ABS_MIN_DB,
        max_passband_ripple_factor=BPF_MAX_PASSBAND_RIPPLE_FACTOR,
        min_passband_ripple_db=BPF_MIN_PASSBAND_RIPPLE_DB,
        min_s11_db=BPF_MIN_S11_DB,
        max_passband_loss_db=BPF_MAX_PASSBAND_LOSS_DB,
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
    reject_reasons: Dict[str, int] = {}  # 统计各原因被拒数量
    for item in dedup.values():
        if _passes_quality(item, cfg, bpf_cfg):
            filtered.append(item)

    # 修正 La_target: 用实际测量的阻带衰减替换理论目标值
    # 这确保 SFT 训练数据中用户问题和模型输出一致反映真实性能
    la_corrected = 0
    for item in filtered:
        metrics = item.get("metrics", {})
        params = item.get("design", {}).get("params", {})
        stop_val = metrics.get("S21_stopband_max_dB")
        if _is_number(stop_val):
            actual_la = round(abs(stop_val), 2)
            original_la = float(params.get("La_target", 0))
            if abs(actual_la - original_la) > 0.5:  # 差异 > 0.5 dB 才修正
                params["La_target_original"] = original_la
                params["La_target"] = actual_la
                la_corrected += 1

    # Assign global IDs
    for idx, item in enumerate(filtered):
        ftype = item.get("design", {}).get("filter_type", "chebyshev")
        band = _get_filter_band(item)
        item["dataset_id"] = f"{ftype}_{band}_{idx:06d}"

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
                "filter_band",
                "order",
                "fc_hz",
                "fs_hz",
                "f_center_hz",
                "bandwidth_hz",
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
            band = _get_filter_band(item)
            writer.writerow(
                {
                    "dataset_id": item.get("dataset_id"),
                    "design_id": item.get("design_id"),
                    "filter_type": design.get("filter_type", "chebyshev"),
                    "filter_band": band,
                    "order": design.get("N"),
                    "fc_hz": params.get("fc") if band != "bandpass" else "",
                    "fs_hz": params.get("fs") if band != "bandpass" else "",
                    "f_center_hz": params.get("f_center") if band == "bandpass" else "",
                    "bandwidth_hz": params.get("bandwidth") if band == "bandpass" else "",
                    "R0": params.get("R0"),
                    "ripple_db": params.get("ripple_db"),
                    "La_target": params.get("La_target"),
                    "S11_max_dB": metrics.get("S11_max_dB"),
                    "S21_passband_min_dB": metrics.get("S21_passband_min_dB"),
                    "S21_stopband_max_dB": metrics.get("S21_stopband_max_dB"),
                    "passband_ripple_dB": metrics.get("passband_ripple_dB"),
                }
            )

    # 分频带统计
    from collections import Counter
    band_raw = Counter(_get_filter_band(item) for item in raw)
    band_dedup = Counter(_get_filter_band(item) for item in dedup.values())
    band_clean = Counter(_get_filter_band(item) for item in filtered)

    print(f"\n{'='*60}")
    print(f"数据清洗报告")
    print(f"{'='*60}")
    print(f"原始样本数: {len(raw)}")
    print(f"去重后样本数: {len(dedup)}")
    print(f"质量筛选后样本数: {len(filtered)}")
    if len(raw) > 0:
        print(f"去重保留率: {len(dedup) / len(raw) * 100:.1f}%")
        print(f"筛选保留率(相对去重后): {len(filtered) / len(dedup) * 100:.1f}%")
        print(f"筛选保留率(相对原始): {len(filtered) / len(raw) * 100:.1f}%")
    print(f"La_target 修正数: {la_corrected}/{len(filtered)} "
          f"({la_corrected/len(filtered)*100:.1f}%)" if filtered else "")

    print(f"\n{'─'*60}")
    print(f"{'频带':<12} {'原始':>8} {'去重后':>8} {'清洗后':>8} {'保留率':>8}")
    print(f"{'─'*60}")
    for band in ['lowpass', 'highpass', 'bandpass']:
        r = band_raw.get(band, 0)
        d = band_dedup.get(band, 0)
        c = band_clean.get(band, 0)
        rate = f"{c / d * 100:.1f}%" if d > 0 else "N/A"
        print(f"{band:<12} {r:>8} {d:>8} {c:>8} {rate:>8}")
    print(f"{'─'*60}")
    print(f"{'合计':<12} {len(raw):>8} {len(dedup):>8} {len(filtered):>8} "
          f"{len(filtered) / len(dedup) * 100:.1f}%" if len(dedup) > 0 else "")

    # 清洗后质量统计
    if filtered:
        s11_vals = [item['metrics']['S11_max_dB'] for item in filtered]
        s21_vals = [item['metrics']['S21_passband_min_dB'] for item in filtered]
        rip_vals = [item['metrics']['passband_ripple_dB'] for item in filtered]
        print(f"\n清洗后数据质量:")
        print(f"  S11 范围: [{min(s11_vals):.1f}, {max(s11_vals):.1f}] dB")
        print(f"  S21_min 范围: [{min(s21_vals):.1f}, {max(s21_vals):.1f}] dB")
        print(f"  Ripple 范围: [{min(rip_vals):.2f}, {max(rip_vals):.2f}] dB")

    print(f"\nJSON 输出: {output_json}")
    print(f"CSV 输出: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
