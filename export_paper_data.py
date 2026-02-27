#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出论文中所有表格与图表的原始数据为 CSV 文件到 paper_materials/ 下
生成:
  1. table1_dataset_statistics.csv     — 数据集统计 (Table I)
  2. table2_ablation_study.csv         — 三方消融实验 (Table II)
  3. table3_comparison.csv             — 对比表格 (Table III)
  4. sparam_lpf_N5.csv                 — S参数 LPF N=5 曲线原始数据
  5. sparam_lpf_N3.csv                 — S参数 LPF N=3 曲线原始数据
  6. sparam_bpf_N7.csv                 — S参数 BPF N=7 曲线原始数据
  7. sparam_bpf_N4.csv                 — S参数 BPF N=4 曲线原始数据
  8. sparam_order_impact.csv           — 阶数影响: N=2~9 各阶S21曲线
  9. training_loss_curve.csv           — 训练loss曲线
 10. ablation_bar_chart.csv            — 消融实验柱状图数据
"""

import csv
import json
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "paper_materials"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Table I — Dataset Statistics
# ============================================================
def export_dataset_statistics():
    meta_path = Path(__file__).parent / "LLaMA-Factory" / "data" / "filter_sft_zhmix" / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}

    rows = [
        ["Split", "Samples", "LPF%", "HPF%", "BPF%", "ZH:EN"],
        ["Train", 25168, "40.3%", "25.7%", "30.1%", "82.6:17.4"],
        ["Val",   1415,  "37.8%", "28.9%", "30.1%", "82.5:17.5"],
        ["Test",  1372,  "41.9%", "29.4%", "24.7%", "83.3:16.7"],
        ["Total", 27955, "",      "",      "",      ""],
    ]
    out = OUT_DIR / "table1_dataset_statistics.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out.name}")


# ============================================================
# 2. Table II — Three-Way Ablation Study
# ============================================================
def export_ablation_study():
    rows = [
        ["Metric", "No_FT", "QLoRA_Early", "RF_FilterLLM"],
        ["JSON Parse Rate", "100%", "100%", "100%"],
        ["Filter Type Accuracy", "100%", "95.4%", "100%"],
        ["Filter Band Accuracy", "100%", "100%", "100%"],
        ["Order Acc (overall)", "19.5%", "12.6%", "83.4%"],
        ["Order Acc LPF", "26.3%", "7.9%", "98.6%"],
        ["Order Acc HPF", "23.3%", "10.0%", "98.2%"],
        ["Order Acc BPF", "0.0%", "26.3%", "47.1%"],
        ["Follow-up Ask Rate", "N/A", "0.0%", "25.0%"],
        ["Multi-turn Order Acc", "N/A", "0.0%", "75.0%"],
    ]
    out = OUT_DIR / "table2_ablation_study.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out.name}")


# ============================================================
# 3. Table III — Comparison with Related Work
# ============================================================
def export_comparison():
    rows = [
        ["Method", "Fine-tuned", "Bands", "Acc_N(%)", "Local_Deploy", "Closed_Loop", "Training_Cost"],
        ["WiseEDA", "No", "LPF", "N/R", "No", "PSO", "API"],
        ["ChipChat", "No", "N/A", "N/A", "No", "No", "API"],
        ["ChipGPT", "No", "N/A", "N/A", "No", "No", "API"],
        ["Baseline(Qwen3-8B)", "No", "3", "19.5", "Yes", "No", "-"],
        ["RF-FilterLLM(Ours)", "Yes", "3", "83.4", "Yes", "Reflective", "9.4h"],
    ]
    out = OUT_DIR / "table3_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out.name}")


# ============================================================
# ABCD Matrix S-parameter Computation (from generate_sparam_figures.py)
# ============================================================
def compute_chebyshev_gk(ripple_db, N):
    pi = np.pi
    beta = np.log(1 / np.tanh(ripple_db / 17.37))
    gamma = np.sinh(beta / (2 * N))
    ak, bk, gk = [], [], []
    for k in range(1, N + 1):
        ak.append(np.sin(((2 * k - 1) * pi) / (2 * N)))
        bk.append(gamma ** 2 + (np.sin(k * pi / N)) ** 2)
    for k in range(1, N + 1):
        if k == 1:
            gk.append(2 * ak[0] / gamma)
        else:
            gk.append((4 * ak[k - 2] * ak[k - 1]) / (bk[k - 2] * gk[k - 2]))
    return gk


def compute_g_load(ripple_db, N):
    ep2 = 10 ** (ripple_db / 10) - 1
    ep = np.sqrt(ep2) if ep2 > 0 else 0
    if N % 2 == 1 or ep == 0:
        return 1.0
    else:
        return (ep + np.sqrt(1 + ep ** 2)) ** 2


def ladder_s21_s11(gk, g_load, omega_norm):
    j = 1j
    s = j * omega_norm
    A, B, C, D = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j
    for k, g in enumerate(gk):
        if k % 2 == 0:
            Z = s * g
            A_n = A + C * Z
            B_n = B + D * Z
            A, B = A_n, B_n
        else:
            Y = s * g
            C_n = C + A * Y
            D_n = D + B * Y
            C, D = C_n, D_n
    R_L = g_load
    denom = A * R_L + B + C * R_L + D
    s21 = 2.0 * np.sqrt(R_L) / denom
    s11 = (A * R_L + B - C * R_L - D) / denom
    return abs(s21) ** 2, abs(s11) ** 2


def compute_lpf_curve(ripple_db, N, fc, f_max, num_points=500):
    gk = compute_chebyshev_gk(ripple_db, N)
    g_load = compute_g_load(ripple_db, N)
    freqs = np.linspace(0.01 * fc, f_max, num_points)
    omega_norm = freqs / fc
    s21_db, s11_db = [], []
    for w in omega_norm:
        s21_sq, s11_sq = ladder_s21_s11(gk, g_load, w)
        s21_db.append(10 * np.log10(max(s21_sq, 1e-20)))
        s11_db.append(10 * np.log10(max(s11_sq, 1e-20)))
    return freqs, np.array(s21_db), np.array(s11_db)


def compute_bpf_curve(ripple_db, N, f0, bw, f_max, num_points=500):
    gk = compute_chebyshev_gk(ripple_db, N)
    g_load = compute_g_load(ripple_db, N)
    freqs = np.linspace(0.01 * f0, f_max, num_points)
    delta = bw / f0
    omega_lp = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        omega_lp[i] = (1.0 / delta) * (f / f0 - f0 / f)
    s21_db, s11_db = [], []
    for w in omega_lp:
        s21_sq, s11_sq = ladder_s21_s11(gk, g_load, abs(w))
        s21_db.append(10 * np.log10(max(s21_sq, 1e-20)))
        s11_db.append(10 * np.log10(max(s11_sq, 1e-20)))
    return freqs, np.array(s21_db), np.array(s11_db)


# ============================================================
# 4-5. S-parameter LPF comparison curves
# ============================================================
def export_sparam_lpf():
    fc, f_max, ripple = 1e9, 4e9, 0.1
    for N in [3, 5]:
        freqs, s21, s11 = compute_lpf_curve(ripple, N, fc, f_max, 500)
        out = OUT_DIR / f"sparam_lpf_N{N}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Freq_Hz", "Freq_GHz", "S21_dB", "S11_dB"])
            for i in range(len(freqs)):
                w.writerow([f"{freqs[i]:.0f}", f"{freqs[i]/1e9:.4f}", f"{s21[i]:.4f}", f"{s11[i]:.4f}"])
        print(f"  ✓ {out.name}  (N={N}, fc={fc/1e9}GHz, ripple={ripple}dB)")
        # Print key value  at fs
        idx_fs = np.argmin(np.abs(freqs - 2e9))
        print(f"    S21 @ fs=2GHz : {s21[idx_fs]:.2f} dB")


# ============================================================
# 6-7. S-parameter BPF comparison curves
# ============================================================
def export_sparam_bpf():
    f0, bw, ripple, f_max = 1.5e9, 200e6, 0.1, 3e9
    for N in [4, 7]:
        freqs, s21, s11 = compute_bpf_curve(ripple, N, f0, bw, f_max, 500)
        out = OUT_DIR / f"sparam_bpf_N{N}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Freq_Hz", "Freq_GHz", "S21_dB", "S11_dB"])
            for i in range(len(freqs)):
                w.writerow([f"{freqs[i]:.0f}", f"{freqs[i]/1e9:.4f}", f"{s21[i]:.4f}", f"{s11[i]:.4f}"])
        print(f"  ✓ {out.name}  (N={N}, f0={f0/1e9}GHz, BW={bw/1e6}MHz)")


# ============================================================
# 8. Order impact: N=2~9 S21 curves for LPF
# ============================================================
def export_order_impact():
    fc, f_max, ripple = 1e9, 4e9, 0.1
    num_points = 500
    freqs = np.linspace(0.01 * fc, f_max, num_points)

    header = ["Freq_Hz", "Freq_GHz"]
    data_cols = {}
    for N in range(2, 10):
        gk = compute_chebyshev_gk(ripple, N)
        g_load = compute_g_load(ripple, N)
        omega_norm = freqs / fc
        s21_list = []
        for w in omega_norm:
            s21_sq, _ = ladder_s21_s11(gk, g_load, w)
            s21_list.append(10 * np.log10(max(s21_sq, 1e-20)))
        data_cols[N] = s21_list
        header.append(f"S21_N{N}_dB")

    out = OUT_DIR / "sparam_order_impact.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(num_points):
            row = [f"{freqs[i]:.0f}", f"{freqs[i]/1e9:.4f}"]
            for N in range(2, 10):
                row.append(f"{data_cols[N][i]:.4f}")
            w.writerow(row)
    print(f"  ✓ {out.name}  (N=2~9, fc=1GHz, ripple=0.1dB)")


# ============================================================
# 9. Training loss curve
# ============================================================
def export_training_loss():
    log_path = Path(__file__).parent / "LLaMA-Factory" / "saves" / "Qwen3-8B-Base" / "lora" / "train_cleaned_v2" / "trainer_log.jsonl"
    if not log_path.exists():
        print(f"  ✗ trainer_log.jsonl 不存在: {log_path}")
        return

    rows = [["step", "loss", "learning_rate", "epoch", "elapsed_time"]]
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "current_steps" in entry:
                rows.append([
                    entry.get("current_steps", ""),
                    entry.get("loss", ""),
                    entry.get("learning_rate", ""),
                    entry.get("epoch", ""),
                    entry.get("elapsed_time", ""),
                ])

    out = OUT_DIR / "training_loss_curve.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out.name}  ({len(rows)-1} data points)")


# ============================================================
# 10. Ablation bar chart data
# ============================================================
def export_ablation_bar():
    rows = [
        ["Model", "LPF_order_acc", "HPF_order_acc", "BPF_order_acc", "Overall_order_acc",
         "Followup_ask_rate", "Multi_turn_order_acc"],
        ["No_FT (Qwen3-8B)", 26.3, 23.3, 0.0, 19.5, "", ""],
        ["QLoRA-Early (11k uncleaned)", 7.9, 10.0, 26.3, 12.6, 0.0, 0.0],
        ["RF-FilterLLM (25k augmented)", 98.6, 98.2, 47.1, 83.4, 25.0, 75.0],
    ]
    out = OUT_DIR / "ablation_bar_chart.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out.name}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("导出论文数据到 paper_materials/")
    print("=" * 60)

    print("\n[表格数据]")
    export_dataset_statistics()
    export_ablation_study()
    export_comparison()

    print("\n[S参数曲线原始数据]")
    export_sparam_lpf()
    export_sparam_bpf()
    export_order_impact()

    print("\n[训练曲线]")
    export_training_loss()

    print("\n[消融图表数据]")
    export_ablation_bar()

    print("\n" + "=" * 60)
    print("所有 CSV 文件已导出到 paper_materials/ 文件夹")
    print("=" * 60)
