r"""
灵敏度分析模块 (Sensitivity Analysis)
======================================

创新点实现：基于灵敏度分析的智能调参策略

核心思想：
  传统的优化方法（GA/PSO）将所有元件参数等权对待，盲目搜索。
  本模块通过计算每个元件对关键性能指标（S11/S21/通带纹波/阻带衰减）的灵敏度，
  告知 LLM "哪些参数的微小变化对性能影响最大"，从而实现：
  1. 让 LLM 优先调整高灵敏度参数 → 收敛更快
  2. 为 SFT 数据中注入灵敏度信息 → 模型学会"射频直觉"

使用方法：
  from sensitivity_analysis import SensitivityAnalyzer
  analyzer = SensitivityAnalyzer()
  report = analyzer.analyze_filter(design_params)
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from adsapi.filter_designer import (
    ChebyshevFilterDesigner,
    ButterworthFilterDesigner,
    get_filter_designer,
)


class SensitivityAnalyzer:
    """
    滤波器灵敏度分析器

    对每个元件值施加微扰 (±δ%)，通过公式（而非 ADS 仿真）
    计算性能指标的变化率，生成灵敏度报告。
    """

    def __init__(self, delta_pct: float = 1.0):
        """
        Args:
            delta_pct: 微扰百分比, 默认 1%
        """
        self.delta_pct = delta_pct

    @staticmethod
    def _ladder_s21(gk: List[float], g_load: float, omega_norm: float) -> float:
        """
        用 ABCD 矩阵法计算归一化低通原型梯形网络在 omega_norm 处的 |S21|^2.

        拓扑: 源阻 g0=1, 第1个元件为串联电感, 第2个为并联电容, 交替...
        gk = [g1, g2, ..., gN], g_load = g_{N+1}

        Returns:
            |S21|^2 (功率传输系数)
        """
        j = 1j
        s = j * omega_norm

        # ABCD 矩阵初始化为单位矩阵
        A, B, C, D = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j

        for k, g in enumerate(gk):
            if k % 2 == 0:
                # 奇数阶 (k=0,2,4,...) → 串联电感, Z = s * g
                Z = s * g
                # ABCD 乘以 [[1, Z], [0, 1]]
                A_new = A + C * Z
                B_new = B + D * Z
                A, B = A_new, B_new
            else:
                # 偶数阶 (k=1,3,5,...) → 并联电容, Y = s * g
                Y = s * g
                # ABCD 乘以 [[1, 0], [Y, 1]]
                C_new = C + A * Y
                D_new = D + B * Y
                C, D = C_new, D_new

        R_L = g_load
        R_S = 1.0  # 归一化源阻
        denom = A + B / R_L + C * R_S * R_L + D * R_S / R_L  # 修正: 实际为 A*R_L + B + C*R_S*R_L + D*R_S
        # 正确公式: S21 = 2*sqrt(R_S*R_L) / (A*R_L + B + C*R_S*R_L + D*R_S)
        # 但归一化时 R_S = 1, 所以:
        denom = A * R_L + B + C * R_L + D
        s21 = 2.0 * np.sqrt(R_L) / denom
        return float(abs(s21) ** 2)

    def _compute_lpf_response(
        self,
        gk: List[float],
        N: int,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
    ) -> Dict[str, float]:
        """
        基于 ABCD 矩阵法计算 LPF 关键指标 (依赖实际 gk 值)
        """
        # gk 列表: [g1, g2, ..., gN]，g_{N+1} 为负载
        # 最后一个 gk 之后的 g_{N+1} 需要从设计中获取
        # Chebyshev N 为奇数时 g_{N+1}=1, 偶数时 g_{N+1} = ε + sqrt(1+ε²) 的平方
        # 简化: 对于大多数实用滤波器, g_{N+1} ≈ 1 (奇数阶) 或按公式计算
        ep2 = 10 ** (ripple_db / 10) - 1
        ep = np.sqrt(ep2) if ep2 > 0 else 0
        if N % 2 == 1 or ep == 0:
            g_load = 1.0
        else:
            g_load = (ep + np.sqrt(1 + ep ** 2)) ** 2

        omega_s = fs / fc  # 归一化阻带频率

        # 阻带衰减: |S21(ωs)|
        s21_sq_stop = self._ladder_s21(gk, g_load, omega_s)
        if s21_sq_stop > 0:
            atten = -10 * np.log10(s21_sq_stop)
        else:
            atten = 100.0

        # 通带内最大纹波: 扫描通带，找 |S21| 最小值
        freqs_pb = np.linspace(0.01, 1.0, 100)
        s21_sq_pb = [self._ladder_s21(gk, g_load, f) for f in freqs_pb]
        min_s21_sq = min(s21_sq_pb) if s21_sq_pb else 1.0
        if min_s21_sq > 0 and min_s21_sq < 1.0:
            passband_ripple = -10 * np.log10(min_s21_sq)
        else:
            passband_ripple = 0.0

        # S11 最大值 (通带): |S11|² = 1 - |S21|²
        s11_sq_max = 1.0 - min_s21_sq
        if s11_sq_max > 0:
            s11_max_dB = 10 * np.log10(s11_sq_max)
        else:
            s11_max_dB = -40.0

        return {
            "stopband_atten_dB": round(float(atten), 3),
            "S11_max_dB": round(float(s11_max_dB), 3),
            "passband_ripple_dB": round(float(passband_ripple), 4),
        }

    def _perturb_element(
        self,
        design: Dict,
        element_type: str,
        index: int,
        delta_pct: float,
    ) -> Dict:
        """
        对设计中某个元件施加微扰.

        Args:
            design: 原始设计字典
            element_type: 'L' 或 'C'
            index: 元件索引 (0-based)
            delta_pct: 微扰百分比
        """
        perturbed = json.loads(json.dumps(design))  # deep copy
        key = element_type  # 'L' or 'C'
        if key in perturbed:
            original_val = perturbed[key][index]
            perturbed[key][index] = original_val * (1 + delta_pct / 100.0)
        return perturbed

    def analyze_lpf(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
        filter_type: str = "chebyshev",
    ) -> Dict[str, Any]:
        """
        对 LPF 进行完整灵敏度分析。

        Returns:
            {
              "baseline": {...},
              "sensitivities": [
                {"element": "L1", "type": "inductor", "value_nH": ...,
                 "sensitivity": {"stopband_atten": ..., "S11_max": ..., "ripple": ...},
                 "rank": 1, "priority": "high"},
                ...
              ],
              "recommendation": "..."
            }
        """
        designer = get_filter_designer(filter_type)
        design = designer.design_lpf_by_attenuation(
            ripple_db=ripple_db, fc=fc, fs=fs, R0=R0, La=La, order=order
        )

        N = design['N']
        gk = design['gk']
        L_vals = design['L']
        C_vals = design['C']

        # 基线指标
        baseline = self._compute_lpf_response(gk, N, ripple_db, fc, fs, R0)

        sensitivities = []

        # 对每个电感扰动
        for i, l_val in enumerate(L_vals):
            # 扰动影响: 改变 L 等效于改变 gk 的实际映射
            # 简化: L_new = L_old * (1+δ) → 等效 gk_new = gk_old * (1+δ)
            gk_perturbed = list(gk)
            k_index = i * 2  # L 对应奇数阶 gk[0], gk[2], ...
            if k_index < len(gk_perturbed):
                gk_perturbed[k_index] *= (1 + self.delta_pct / 100.0)

            perturbed_metrics = self._compute_lpf_response(
                gk_perturbed, N, ripple_db, fc, fs, R0
            )

            sens = {}
            for metric in baseline:
                if baseline[metric] != 0:
                    sens[metric] = round(
                        (perturbed_metrics[metric] - baseline[metric])
                        / (baseline[metric] * self.delta_pct / 100.0),
                        4,
                    )
                else:
                    sens[metric] = 0.0

            sensitivities.append({
                "element": f"L{i+1}",
                "type": "inductor",
                "value_nH": l_val,
                "sensitivity": sens,
                "abs_impact": sum(abs(v) for v in sens.values()),
            })

        # 对每个电容扰动
        for i, c_val in enumerate(C_vals):
            gk_perturbed = list(gk)
            k_index = i * 2 + 1  # C 对应偶数阶 gk[1], gk[3], ...
            if k_index < len(gk_perturbed):
                gk_perturbed[k_index] *= (1 + self.delta_pct / 100.0)

            perturbed_metrics = self._compute_lpf_response(
                gk_perturbed, N, ripple_db, fc, fs, R0
            )

            sens = {}
            for metric in baseline:
                if baseline[metric] != 0:
                    sens[metric] = round(
                        (perturbed_metrics[metric] - baseline[metric])
                        / (baseline[metric] * self.delta_pct / 100.0),
                        4,
                    )
                else:
                    sens[metric] = 0.0

            sensitivities.append({
                "element": f"C{i+1}",
                "type": "capacitor",
                "value_pF": c_val,
                "sensitivity": sens,
                "abs_impact": sum(abs(v) for v in sens.values()),
            })

        # 排名
        sensitivities.sort(key=lambda x: x["abs_impact"], reverse=True)
        for rank, s in enumerate(sensitivities, 1):
            s["rank"] = rank
            if rank <= max(1, len(sensitivities) // 3):
                s["priority"] = "high"
            elif rank <= max(2, len(sensitivities) * 2 // 3):
                s["priority"] = "medium"
            else:
                s["priority"] = "low"

        # 生成自然语言建议
        top = sensitivities[0] if sensitivities else None
        recommendation = ""
        if top:
            recommendation = (
                f"灵敏度最高的元件是 {top['element']}（{top['type']}, "
                f"{'值=' + str(top.get('value_nH', top.get('value_pF')))} "
                f"{'nH' if top['type'] == 'inductor' else 'pF'}），"
                f"综合影响指数={top['abs_impact']:.4f}。"
                f"建议优先调整该元件以加速收敛。"
            )

        return {
            "design": {
                "filter_type": filter_type,
                "N": N,
                "L": L_vals,
                "C": C_vals,
                "params": design['params'],
            },
            "baseline_metrics": baseline,
            "sensitivities": sensitivities,
            "recommendation": recommendation,
            "delta_pct": self.delta_pct,
        }

    def generate_sensitivity_sft_sample(
        self,
        ripple_db: float,
        fc: float,
        fs: float,
        R0: float,
        La: float,
        order: Optional[int] = None,
        filter_type: str = "chebyshev",
    ) -> Optional[Dict]:
        """
        生成包含灵敏度信息的 SFT 训练样本。

        格式:
          user: "分析以下滤波器的灵敏度: ..."
          assistant: JSON 灵敏度报告
        """
        try:
            report = self.analyze_lpf(
                ripple_db=ripple_db, fc=fc, fs=fs, R0=R0, La=La,
                order=order, filter_type=filter_type,
            )
        except Exception:
            return None

        user_msg = (
            f"请分析以下 {filter_type} 低通滤波器的元件灵敏度：\n"
            f"fc={fc} Hz, fs={fs} Hz, ripple_db={ripple_db} dB, "
            f"R0={R0} ohm, La_target={La} dB, order={report['design']['N']}。\n"
            f"请告诉我哪些元件对性能影响最大，应该优先调整。"
        )

        # 精简输出
        compact_sens = []
        for s in report["sensitivities"]:
            compact_sens.append({
                "element": s["element"],
                "priority": s["priority"],
                "impact": round(s["abs_impact"], 4),
            })

        assistant_msg = json.dumps({
            "baseline": report["baseline_metrics"],
            "sensitivity_ranking": compact_sens,
            "recommendation": report["recommendation"],
        }, ensure_ascii=False)

        return {
            "messages": [
                {"role": "system", "content": "你是射频滤波器设计专家，擅长元件灵敏度分析。"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }


def generate_sensitivity_dataset(
    num_samples: int = 200,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    批量生成灵敏度分析 SFT 数据。
    """
    import random

    rng = random.Random(seed)
    np.random.seed(seed)
    analyzer = SensitivityAnalyzer(delta_pct=1.0)

    samples = []
    for i in range(num_samples):
        ripple = rng.choice([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        fc = rng.choice([400e6, 600e6, 800e6, 1e9, 1.2e9, 1.5e9, 2e9])
        fs_ratio = rng.uniform(1.3, 3.0)
        fs = fc * fs_ratio
        La = rng.uniform(25, 60)
        R0 = rng.choice([25, 50, 75, 100])
        order = rng.randint(3, 9)
        ft = rng.choice(["chebyshev", "butterworth"])

        sample = analyzer.generate_sensitivity_sft_sample(
            ripple_db=ripple, fc=fc, fs=fs, R0=R0, La=La,
            order=order, filter_type=ft,
        )
        if sample:
            samples.append(sample)

        if (i + 1) % 50 == 0:
            print(f"  灵敏度样本: {i+1}/{num_samples}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"灵敏度数据集已保存: {output_path} ({len(samples)} 条)")

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="灵敏度分析数据集生成")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reflection_dataset/sensitivity_train.jsonl")
    args = parser.parse_args()

    samples = generate_sensitivity_dataset(
        num_samples=args.num,
        seed=args.seed,
        output_path=Path(args.output),
    )
    print(f"\n生成灵敏度样本: {len(samples)} 条")
