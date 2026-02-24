r"""
课程学习 (Curriculum Learning) 数据排序模块
=============================================

创新点实现：对 SFT 训练数据按难度排序

核心思想：
  人类学习射频设计是从简单到复杂递进的。让模型也遵循这一规律：
  1. 先学低阶(3阶)、标准参数(R0=50, fc=1GHz)的简单样本
  2. 再学高阶(7-9阶)、极端参数的复杂样本
  3. 最后学多轮对话、反思修正等高难度样本

  难度评分公式：
    difficulty = w1 * order_score + w2 * param_extremity + w3 * conversation_complexity

使用方法：
    python curriculum_learning.py --input train.jsonl --output train_sorted.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_order_from_messages(messages: List[Dict]) -> int:
    """从对话中提取滤波器阶数"""
    for msg in messages:
        content = msg.get("content", "")
        # 尝试从 JSON 中提取
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "order" in data:
                return int(data["order"])
        except (json.JSONDecodeError, ValueError):
            pass
        # 尝试正则
        m = re.search(r'"order"\s*:\s*(\d+)', content)
        if m:
            return int(m.group(1))
    return 5  # 默认


def _extract_params_from_messages(messages: List[Dict]) -> Dict[str, float]:
    """从对话中提取参数"""
    for msg in messages:
        content = msg.get("content", "")
        try:
            data = json.loads(content)
            if isinstance(data, dict) and ("fc" in data or "f_center" in data):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _compute_difficulty(sample: Dict) -> float:
    """
    计算样本难度分数 (0-1)

    因子：
      1. 阶数 (order): 高阶更难
      2. 参数极端性: 非标准 R0, 频率范围边缘
      3. 对话复杂度: 多轮 > 单轮, 反思 > 普通
      4. 滤波器类型: BPF > HPF > LPF
    """
    messages = sample.get("messages", [])
    num_turns = len([m for m in messages if m.get("role") in ("user", "assistant")])

    order = _extract_order_from_messages(messages)
    params = _extract_params_from_messages(messages)

    # 1. 阶数评分 (0-1): 3阶=0, 9阶=1
    order_score = min(1.0, max(0.0, (order - 3) / 6.0))

    # 2. 参数极端性 (0-1)
    extremity = 0.0
    R0 = params.get("R0", 50)
    if R0 != 50:
        extremity += 0.2
    fc = params.get("fc", params.get("f_center", 1e9))
    if fc > 3e9 or fc < 400e6:
        extremity += 0.2
    ripple = params.get("ripple_db", 0.1)
    if ripple > 0.5 or ripple < 0.05:
        extremity += 0.1
    la = params.get("La_target", params.get("La", 30))
    if la > 50 or la < 20:
        extremity += 0.15
    extremity = min(1.0, extremity)

    # 3. 对话复杂度 (0-1)
    conv_complexity = 0.0
    if num_turns >= 4:
        conv_complexity = 0.5  # 多轮对话
    if num_turns >= 6:
        conv_complexity = 0.8
    # 检查是否包含反思关键词
    full_text = " ".join(m.get("content", "") for m in messages)
    if "反思" in full_text or "修正" in full_text or "调整" in full_text or "未达标" in full_text:
        conv_complexity = max(conv_complexity, 0.9)
    if "灵敏度" in full_text or "sensitivity" in full_text.lower():
        conv_complexity = max(conv_complexity, 0.85)

    # 4. 滤波器类型
    type_score = 0.0
    if "bandpass" in full_text or "带通" in full_text or "BPF" in full_text:
        type_score = 0.3
    elif "highpass" in full_text or "高通" in full_text or "HPF" in full_text:
        type_score = 0.15

    # 加权求和
    difficulty = (
        0.25 * order_score
        + 0.20 * extremity
        + 0.35 * conv_complexity
        + 0.20 * type_score
    )

    return round(difficulty, 4)


def sort_by_curriculum(
    input_path: Path,
    output_path: Path,
    reverse: bool = False,
) -> Dict[str, Any]:
    """
    读取 JSONL 文件，按难度排序输出。

    Args:
        input_path: 输入 JSONL 文件
        output_path: 输出排序后的 JSONL 文件
        reverse: True=从难到易 (anti-curriculum)

    Returns:
        排序统计信息
    """
    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            diff = _compute_difficulty(sample)
            samples.append({"data": sample, "difficulty": diff})

    # 排序
    samples.sort(key=lambda x: x["difficulty"], reverse=reverse)

    # 写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s["data"], ensure_ascii=False) + "\n")

    # 统计
    difficulties = [s["difficulty"] for s in samples]
    stats = {
        "total": len(samples),
        "min_difficulty": min(difficulties) if difficulties else 0,
        "max_difficulty": max(difficulties) if difficulties else 0,
        "mean_difficulty": round(sum(difficulties) / len(difficulties), 4) if difficulties else 0,
        "easy_count": sum(1 for d in difficulties if d < 0.3),
        "medium_count": sum(1 for d in difficulties if 0.3 <= d < 0.6),
        "hard_count": sum(1 for d in difficulties if d >= 0.6),
    }

    print(f"课程学习排序完成")
    print(f"  总样本: {stats['total']}")
    print(f"  难度范围: [{stats['min_difficulty']}, {stats['max_difficulty']}]")
    print(f"  平均难度: {stats['mean_difficulty']}")
    print(f"  简单/中等/困难: {stats['easy_count']}/{stats['medium_count']}/{stats['hard_count']}")
    print(f"  输出: {output_path}")

    return stats


def merge_and_sort_datasets(
    paths: List[Path],
    output_path: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    合并多个 JSONL 数据集，并按课程学习难度排序。

    适用于将 reflection + sensitivity + normal SFT 数据合并为一个有序训练集。
    """
    import random

    all_samples = []
    for p in paths:
        if not p.exists():
            print(f"  警告: {p} 不存在，跳过")
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    diff = _compute_difficulty(sample)
                    all_samples.append({"data": sample, "difficulty": diff})
        print(f"  已读取: {p.name}")

    # 按难度排序 (从易到难)
    all_samples.sort(key=lambda x: x["difficulty"])

    # 在同等难度内小幅打乱 (防止过度有序)
    rng = random.Random(seed)
    bucket_size = max(1, len(all_samples) // 20)
    for i in range(0, len(all_samples), bucket_size):
        bucket = all_samples[i:i + bucket_size]
        rng.shuffle(bucket)
        all_samples[i:i + bucket_size] = bucket

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s["data"], ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_samples),
        "sources": [str(p) for p in paths],
    }
    print(f"\n合并排序完成: {len(all_samples)} 条 → {output_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="课程学习数据排序")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", type=str, required=True, help="输出排序后 JSONL 路径")
    parser.add_argument("--reverse", action="store_true", help="从难到易排序")
    args = parser.parse_args()

    sort_by_curriculum(
        input_path=Path(args.input),
        output_path=Path(args.output),
        reverse=args.reverse,
    )
