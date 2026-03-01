#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ALL paper figures with unified Arial font and consistent sizing.

Figures:
  1. order_scatter.pdf       — 175-sample scatter (pred vs gt order)
  2. sparam_spec_impact.pdf  — S21 + S11 side-by-side for N=3,4,5,6
  3. sparam_closedloop_iter.pdf — Closed-loop iteration comparison (Iter1 vs Iter2 S-param)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from collections import defaultdict

# ============================================================
# Global: Arial font, consistent size
# ============================================================
# Try to use Arial; fallback to sans-serif
arial_found = False
for f in fm.findSystemFonts():
    if 'arial' in f.lower() and 'bold' not in f.lower() and 'italic' not in f.lower():
        arial_found = True
        break

if arial_found:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
    print("[WARN] Arial not found, using Helvetica/DejaVu Sans fallback")

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Consistent font sizes across all figures
FONT_SIZE_LABEL = 9       # axis labels
FONT_SIZE_TICK = 8        # tick labels
FONT_SIZE_LEGEND = 7      # legend text
FONT_SIZE_ANNOTATION = 7  # annotations

plt.rcParams['font.size'] = FONT_SIZE_TICK
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND

SAVE_DIR = Path(__file__).parent / 'paper_materials' / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper: Chebyshev prototype computation
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


def ladder_s_params(gk, g_load, omega_norm):
    """Compute S21 and S11 via ABCD matrix."""
    j = 1j
    s = j * omega_norm
    A, B, C, D = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j
    for k, g in enumerate(gk):
        if k % 2 == 0:  # series inductor
            Z = s * g
            A, B = A + C * Z, B + D * Z
        else:  # shunt capacitor
            Y = s * g
            C, D = C + A * Y, D + B * Y
    R_L = g_load
    denom = A * R_L + B + C * R_L + D
    s21_sq = abs(2.0 * np.sqrt(R_L) / denom) ** 2
    s11_sq = abs((A * R_L + B - C * R_L - D) / denom) ** 2
    return s21_sq, s11_sq


def compute_lpf_sparam(ripple_db, N, fc, num_points=500):
    """Compute S21 and S11 vs frequency for a Chebyshev LPF."""
    gk = compute_chebyshev_gk(ripple_db, N)
    g_load = compute_g_load(ripple_db, N)
    freqs_hz = np.linspace(0.01 * fc, 4.0 * fc, num_points)
    omega_norm = freqs_hz / fc

    S21_dB, S11_dB = [], []
    for w in omega_norm:
        s21_sq, s11_sq = ladder_s_params(gk, g_load, w)
        S21_dB.append(10 * np.log10(max(s21_sq, 1e-20)))
        S11_dB.append(10 * np.log10(max(s11_sq, 1e-20)))

    return freqs_hz, np.array(S21_dB), np.array(S11_dB)


# ============================================================
# Figure 1: Order Prediction Scatter Plot
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
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = SAVE_DIR / 'order_scatter'
    fig.savefig(str(out) + '.pdf')
    fig.savefig(str(out) + '.png')
    plt.close(fig)
    print(f"  [OK] order_scatter.pdf")


# ============================================================
# Figure 2: S-param Spec Impact — S21 (left) + S11 (right)
# ============================================================
def plot_sparam_spec_impact():
    """
    Chebyshev LPF: fc=1GHz, fs=2GHz, Lr=0.1dB, La=30dB => N_min=5
    Horizontal layout: S21 (left) | S11 (right)
    No title inside figure.
    """
    fc = 1e9
    fs = 2e9
    ripple = 0.1
    La_target = 30

    ks = fs / fc
    eps_sq_r = 10**(ripple/10) - 1
    eps_sq_a = 10**(La_target/10) - 1
    N_exact = np.arccosh(np.sqrt(eps_sq_a / eps_sq_r)) / np.arccosh(ks)
    N_min = int(np.ceil(N_exact))
    print(f"  Spec: La={La_target}dB, N_min={N_min}")

    freqs = np.linspace(0.01e9, 4e9, 1000)
    omega_norm = freqs / fc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

    colors_n = {3: '#FF5722', 4: '#FF9800', 5: '#2196F3', 6: '#9E9E9E'}
    styles_n = {3: '--', 4: '-.', 5: '-', 6: ':'}
    lw_n = {3: 1.2, 4: 1.2, 5: 2.0, 6: 1.2}

    for N in [3, 4, 5, 6]:
        gk = compute_chebyshev_gk(ripple, N)
        g_load = compute_g_load(ripple, N)
        s21_db, s11_db = [], []
        for w in omega_norm:
            s21_sq, s11_sq = ladder_s_params(gk, g_load, w)
            s21_db.append(10 * np.log10(max(s21_sq, 1e-20)))
            s11_db.append(10 * np.log10(max(s11_sq, 1e-20)))
        s21_db = np.array(s21_db)
        s11_db = np.array(s11_db)

        idx_fs = np.argmin(np.abs(freqs - fs))
        atten_fs = -s21_db[idx_fs]

        label = f'$N={N}$'
        if N == N_min:
            label += f' ({atten_fs:.0f} dB)'
        elif N < N_min:
            label += f' ({atten_fs:.0f} dB, FAIL)'

        ax1.plot(freqs / 1e9, s21_db, color=colors_n[N],
                 linestyle=styles_n[N], linewidth=lw_n[N], label=label)
        ax2.plot(freqs / 1e9, s11_db, color=colors_n[N],
                 linestyle=styles_n[N], linewidth=lw_n[N], label=f'$N={N}$')

    # S21 axis
    ax1.axhline(-La_target, color='red', linewidth=1.0, linestyle='-', alpha=0.6)
    ax1.text(3.5, -La_target + 1.5, f'$L_A = {La_target}$ dB', color='red',
             fontsize=FONT_SIZE_ANNOTATION, ha='right', va='bottom')
    ax1.axvline(fc / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.axvline(fs / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.text(fc / 1e9, -55, '$f_c$', fontsize=FONT_SIZE_ANNOTATION, ha='center', color='gray')
    ax1.text(fs / 1e9, -55, '$f_s$', fontsize=FONT_SIZE_ANNOTATION, ha='center', color='gray')
    ax1.axvspan(0, fc / 1e9, alpha=0.05, color='green')
    ax1.axvspan(fs / 1e9, 4, alpha=0.05, color='red')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('$|S_{21}|$ (dB)')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-60, 2)
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, fontsize=FONT_SIZE_LABEL,
             fontweight='bold', va='top', ha='left')

    # S11 axis
    ax2.axhline(-10, color='#FF9800', linewidth=1.0, linestyle='--', alpha=0.7)
    ax2.text(3.5, -9.0, '$-10$ dB', color='#FF9800',
             fontsize=FONT_SIZE_ANNOTATION, ha='right', va='bottom')
    ax2.axvline(fc / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.text(fc / 1e9, -48, '$f_c$', fontsize=FONT_SIZE_ANNOTATION, ha='center', color='gray')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('$|S_{11}|$ (dB)')
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-50, 2)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, fontsize=FONT_SIZE_LABEL,
             fontweight='bold', va='top', ha='left')

    fig.tight_layout()
    out = SAVE_DIR / 'sparam_spec_impact'
    fig.savefig(str(out) + '.pdf')
    fig.savefig(str(out) + '.png')
    plt.close(fig)
    print(f"  [OK] sparam_spec_impact.pdf (S21 + S11 horizontal)")


# ============================================================
# Figure 3: Closed-loop Iteration Comparison (S-param curves)
# ============================================================
def plot_closedloop_comparison():
    """
    Iteration 1 (ripple=0.5dB) vs Iteration 2 (ripple=0.25dB)
    for the same LPF: fc=1GHz, N=5.
    Show S21 + S11 side-by-side with both iterations overlaid.
    No title inside figure — info goes to LaTeX caption.
    """
    fc = 1e9
    N = 5

    # Iteration 1: ripple = 0.5 dB
    ripple1 = 0.5
    freqs1, s21_1, s11_1 = compute_lpf_sparam(ripple1, N, fc, num_points=500)

    # Iteration 2: ripple = 0.25 dB
    ripple2 = 0.25
    freqs2, s21_2, s11_2 = compute_lpf_sparam(ripple2, N, fc, num_points=500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # S21
    ax1.plot(freqs1 / 1e9, s21_1, '--', color='#FF5722', linewidth=1.5,
             label=f'Iter 1 ($L_r$=0.5 dB)')
    ax1.plot(freqs2 / 1e9, s21_2, '-', color='#2196F3', linewidth=1.8,
             label=f'Iter 2 ($L_r$=0.25 dB)')
    ax1.axvline(fc / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.text(fc / 1e9, -55, '$f_c$', fontsize=FONT_SIZE_ANNOTATION, ha='center', color='gray')
    ax1.axvspan(0, fc / 1e9, alpha=0.05, color='green')
    ax1.axvspan(2.0, 4, alpha=0.05, color='red')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('$|S_{21}|$ (dB)')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-60, 2)
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, fontsize=FONT_SIZE_LABEL,
             fontweight='bold', va='top', ha='left')

    # S11
    ax2.plot(freqs1 / 1e9, s11_1, '--', color='#FF5722', linewidth=1.5,
             label=f'Iter 1 ($S_{{11}}$=$-$9.6 dB)')
    ax2.plot(freqs2 / 1e9, s11_2, '-', color='#2196F3', linewidth=1.8,
             label=f'Iter 2 ($S_{{11}}$=$-$12.4 dB)')
    ax2.axhline(-10, color='red', linewidth=1.0, linestyle='--', alpha=0.6)
    ax2.text(0.9, -9.0, '$-$10 dB spec', color='red',
             fontsize=FONT_SIZE_ANNOTATION, ha='left', va='bottom')
    ax2.axvline(fc / 1e9, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('$|S_{11}|$ (dB)')
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-50, 2)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, fontsize=FONT_SIZE_LABEL,
             fontweight='bold', va='top', ha='left')

    fig.tight_layout()
    out = SAVE_DIR / 'sparam_closedloop_iter'
    fig.savefig(str(out) + '.pdf')
    fig.savefig(str(out) + '.png')
    plt.close(fig)
    print(f"  [OK] sparam_closedloop_iter.pdf (Iter 1 vs Iter 2)")


# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 50)
    print("Generating ALL paper figures (Arial, unified)")
    print("=" * 50)

    print(f"\nFont family: {plt.rcParams['font.sans-serif']}")

    print("\n[1] Order prediction scatter plot")
    plot_order_scatter()

    print("\n[2] S-parameter spec impact (S21 + S11)")
    plot_sparam_spec_impact()

    print("\n[3] Closed-loop iteration comparison")
    plot_closedloop_comparison()

    print("\n" + "=" * 50)
    print("Done. All figures in paper_materials/figures/")
