#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成两个更有说服力的论文实验图:

1. order_scatter.pdf — 175个测试样本的 (gt_order, pred_order) 散点图
   按LPF/HPF/BPF着色，展示BPF系统性低估模式

2. sparam_spec_impact.pdf — 给定一个具体spec需求(La_target=40dB),
   展示N=3/4/5/6的S21曲线与spec门限线，
   说明"差一阶就不达标"

3. closed_loop_convergence.pdf — 闭环agent 2次迭代收敛过程图
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

SAVE_DIR = Path(__file__).parent / 'paper_materials' / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Order Prediction Scatter Plot
# ============================================================
def plot_order_scatter():
    eval_path = Path(__file__).parent / 'paper_materials' / 'data' / 'eval_results_200samples.json'
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['details']['full']

    band_data = defaultdict(lambda: {'gt': [], 'pred': []})
    for s in samples:
        band = s['gt']['filter_band']
        gt_order = s['gt']['order']
        pred_order = s['pred'].get('order', 0)
        band_data[band]['gt'].append(gt_order)
        band_data[band]['pred'].append(pred_order)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    colors = {'lowpass': '#2196F3', 'highpass': '#4CAF50', 'bandpass': '#FF5722'}
    labels = {'lowpass': 'LPF', 'highpass': 'HPF', 'bandpass': 'BPF'}
    markers = {'lowpass': 'o', 'highpass': 's', 'bandpass': '^'}

    all_orders = []
    for band in ['lowpass', 'highpass', 'bandpass']:
        d = band_data[band]
        gt = np.array(d['gt'])
        pred = np.array(d['pred'])
        all_orders.extend(gt)
        all_orders.extend(pred)

        correct = gt == pred
        n_correct = correct.sum()
        n_total = len(gt)

        # Add jitter for overlapping points
        jitter_gt = gt + np.random.uniform(-0.15, 0.15, len(gt))
        jitter_pred = pred + np.random.uniform(-0.15, 0.15, len(pred))

        ax.scatter(jitter_gt[correct], jitter_pred[correct],
                   c=colors[band], marker=markers[band], s=30, alpha=0.7,
                   label=f'{labels[band]} ({n_correct}/{n_total})',
                   edgecolors='white', linewidths=0.3)
        ax.scatter(jitter_gt[~correct], jitter_pred[~correct],
                   c=colors[band], marker=markers[band], s=30, alpha=0.35,
                   edgecolors='red', linewidths=1.0)

    min_o = min(all_orders) - 0.5
    max_o = max(all_orders) + 0.5
    ax.plot([min_o, max_o], [min_o, max_o], 'k--', linewidth=0.8, alpha=0.5, label='Ideal')

    ax.set_xlabel('Ground-Truth Order $N_{\\mathrm{gt}}$')
    ax.set_ylabel('Predicted Order $N_{\\mathrm{pred}}$')
    ax.set_xlim(min_o, max_o)
    ax.set_ylim(min_o, max_o)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = SAVE_DIR / 'order_scatter.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"  [OK] {out.name}")

    # Also export CSV
    csv_out = SAVE_DIR.parent / 'order_scatter_data.csv'
    import csv
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'filter_band', 'gt_order', 'pred_order', 'correct'])
        for s in samples:
            band = s['gt']['filter_band']
            gt_o = s['gt']['order']
            pred_o = s['pred'].get('order', 0)
            w.writerow([s['idx'], band, gt_o, pred_o, 1 if gt_o == pred_o else 0])
    print(f"  [OK] {csv_out.name} (CSV)")


# ============================================================
# 2. S-parameter with Spec Lines — "One Order Off = Fail"
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

def ladder_s21(gk, g_load, omega_norm):
    j = 1j
    s = j * omega_norm
    A, B, C, D = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j
    for k, g in enumerate(gk):
        if k % 2 == 0:
            Z = s * g; A, B = A + C * Z, B + D * Z
        else:
            Y = s * g; C, D = C + A * Y, D + B * Y
    R_L = g_load
    denom = A * R_L + B + C * R_L + D
    s21 = 2.0 * np.sqrt(R_L) / denom
    return abs(s21) ** 2


def plot_sparam_spec_impact():
    """
    一个真实测试用例:
    Chebyshev LPF, fc=1GHz, La_target=35dB, ripple=0.1dB
    => 最小阶数 N=5 刚好满足
    展示 N=3,4,5,6 的S21曲线 + spec门限线
    """
    fc = 1e9
    fs = 2e9
    ripple = 0.1
    La_target = 30  # dB — N_min=5 for this spec

    # Verify: with N=5, what attenuation at fs?
    # Chebyshev: N = ceil(acosh(sqrt((10^(La/10)-1)/(10^(Lr/10)-1))) / acosh(ks))
    ks = fs / fc
    eps_sq_r = 10**(ripple/10) - 1
    eps_sq_a = 10**(La_target/10) - 1
    N_exact = np.arccosh(np.sqrt(eps_sq_a / eps_sq_r)) / np.arccosh(ks)
    N_min = int(np.ceil(N_exact))
    print(f"  Spec: fc={fc/1e9}GHz, fs={fs/1e9}GHz, Lr={ripple}dB, La={La_target}dB")
    print(f"  N_exact = {N_exact:.2f}, N_min = {N_min}")

    freqs = np.linspace(0.01e9, 4e9, 1000)
    omega_norm = freqs / fc

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    colors_n = {3: '#FF5722', 4: '#FF9800', 5: '#2196F3', 6: '#9E9E9E'}
    styles_n = {3: '--', 4: '-.', 5: '-', 6: ':'}
    lw_n = {3: 1.2, 4: 1.2, 5: 2.0, 6: 1.2}

    for N in [3, 4, 5, 6]:
        gk = compute_chebyshev_gk(ripple, N)
        g_load = compute_g_load(ripple, N)
        s21_db = []
        for w in omega_norm:
            s21_sq = ladder_s21(gk, g_load, w)
            s21_db.append(10 * np.log10(max(s21_sq, 1e-20)))
        s21_db = np.array(s21_db)

        # Compute attenuation at fs
        idx_fs = np.argmin(np.abs(freqs - fs))
        atten_fs = -s21_db[idx_fs]

        label = f'$N={N}$'
        if N == N_min:
            label += f' (correct, {atten_fs:.0f} dB)'
        elif N < N_min:
            label += f' ({atten_fs:.0f} dB, FAIL)'

        ax.plot(freqs / 1e9, s21_db, color=colors_n[N],
                linestyle=styles_n[N], linewidth=lw_n[N], label=label)

    # Spec lines
    ax.axhline(-La_target, color='red', linewidth=1.0, linestyle='-', alpha=0.6)
    ax.text(3.5, -La_target + 1.5, f'$L_A = {La_target}$ dB', color='red',
            fontsize=7, ha='right', va='bottom')

    ax.axvline(fc / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.axvline(fs / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.text(fc / 1e9, -55, '$f_c$', fontsize=7, ha='center', color='gray')
    ax.text(fs / 1e9, -55, '$f_s$', fontsize=7, ha='center', color='gray')

    # Shade passband
    ax.axvspan(0, fc / 1e9, alpha=0.05, color='green')
    # Shade stopband
    ax.axvspan(fs / 1e9, 4, alpha=0.05, color='red')

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$|S_{21}|$ (dB)')
    ax.set_xlim(0, 4)
    ax.set_ylim(-60, 2)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = SAVE_DIR / 'sparam_spec_impact.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"  [OK] {out.name}")

    # Export CSV
    csv_out = SAVE_DIR.parent / 'sparam_spec_impact_data.csv'
    import csv
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        header = ['Freq_Hz', 'Freq_GHz']
        for N in [3, 4, 5, 6]:
            header.append(f'S21_N{N}_dB')
        w.writerow(header)
        for i in range(len(freqs)):
            row = [f'{freqs[i]:.0f}', f'{freqs[i]/1e9:.4f}']
            for N in [3, 4, 5, 6]:
                gk = compute_chebyshev_gk(ripple, N)
                g_load = compute_g_load(ripple, N)
                s21_sq = ladder_s21(gk, g_load, omega_norm[i])
                row.append(f'{10 * np.log10(max(s21_sq, 1e-20)):.4f}')
            w.writerow(row)
    print(f"  [OK] {csv_out.name} (CSV)")


# ============================================================
# 3. Closed-loop Convergence
# ============================================================
def plot_closed_loop():
    """
    从 llm_ads_outputs/closed_loop_20260224_213458.json 提取迭代数据
    画 2-iteration 收敛柱状图
    """
    json_path = Path(__file__).parent / 'llm_ads_outputs' / 'closed_loop_20260224_213458.json'
    if not json_path.exists():
        # try alternative
        json_path = Path(__file__).parent / 'llm_ads_outputs' / 'closed_loop_20260224_205943.json'
    if not json_path.exists():
        print("  [SKIP] No closed-loop JSON found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        cl = json.load(f)

    iterations = cl.get('iterations', [])
    if len(iterations) < 2:
        print(f"  [SKIP] Only {len(iterations)} iterations")
        return

    # Extract metrics per iteration
    metrics_names = ['$S_{11,\\max}$\n(dB)', '$S_{21}$ ripple\n(dB)', 'Status']
    iter_data = []
    for it in iterations:
        metrics = it.get('metrics', {})
        ev = it.get('eval', {})
        spec = it.get('spec', {})
        s11 = metrics.get('S11_max_dB', it.get('S11_max_dB', 0))
        ripple_actual = metrics.get('passband_ripple_dB', it.get('passband_ripple_dB', 0))
        passed = ev.get('passed', it.get('passed', False))
        iter_data.append({
            's11': s11,
            'ripple': ripple_actual,
            'passed': passed,
            'spec_ripple': spec.get('ripple_db', it.get('spec', {}).get('ripple_db', 0.5)),
            'stopband': metrics.get('S21_stopband_max_dB', 0),
        })

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2))

    # Left: S11 bar chart
    ax = axes[0]
    iters = np.arange(len(iter_data))
    s11_vals = [d['s11'] for d in iter_data]
    colors = ['#FF5722' if not d['passed'] else '#4CAF50' for d in iter_data]
    bars = ax.bar(iters, s11_vals, color=colors, width=0.5, edgecolor='white')
    ax.axhline(-10, color='red', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.text(len(iters)-0.5, -9.5, 'Spec: $-10$ dB', fontsize=6, color='red', ha='right')
    ax.set_ylabel('$S_{11,\\max}$ (dB)', fontsize=8)
    ax.set_xticks(iters)
    ax.set_xticklabels([f'Iter {i+1}' for i in iters], fontsize=7)
    ax.set_title('Passband Matching', fontsize=8)
    for i, v in enumerate(s11_vals):
        ax.text(i, v - 0.3, f'{v:.1f}', ha='center', va='top', fontsize=7, fontweight='bold')

    # Right: Ripple spec change
    ax = axes[1]
    spec_ripple = [d['spec_ripple'] for d in iter_data]
    bars = ax.bar(iters, spec_ripple, color=colors, width=0.5, edgecolor='white')
    ax.set_ylabel('Ripple spec (dB)', fontsize=8)
    ax.set_xticks(iters)
    ax.set_xticklabels([f'Iter {i+1}' for i in iters], fontsize=7)
    ax.set_title('LLM Self-Correction', fontsize=8)
    for i, v in enumerate(spec_ripple):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    out = SAVE_DIR / 'closed_loop_convergence.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 50)
    print("生成论文实验图")
    print("=" * 50)

    print("\n[1] Order prediction scatter plot")
    plot_order_scatter()

    print("\n[2] S-parameter spec impact (N=3,4,5,6)")
    plot_sparam_spec_impact()

    print("\n[3] Closed-loop convergence")
    plot_closed_loop()

    print("\n" + "=" * 50)
    print("Done. Figures saved to paper_materials/figures/")
