#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S参数仿真图 + 阶数影响对比图
用于论文图表：展示模型预测的正确/错误阶数对滤波器性能的影响

生成:
  1. sparam_lpf_comparison.pdf — 低通滤波器：正确阶数 vs 错误阶数的S参数对比
  2. sparam_bpf_comparison.pdf — 带通滤波器S参数对比
  3. order_impact_sparam.pdf  — 不同阶数对S21阻带衰减的影响曲线
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11

SAVE_DIR = Path(__file__).parent / 'paper_materials' / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ===== ABCD matrix based S-parameter computation =====

def compute_chebyshev_gk(ripple_db, N):
    """Compute Chebyshev prototype g-values."""
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
    """Compute load impedance g_{N+1}."""
    ep2 = 10 ** (ripple_db / 10) - 1
    ep = np.sqrt(ep2) if ep2 > 0 else 0
    if N % 2 == 1 or ep == 0:
        return 1.0
    else:
        return (ep + np.sqrt(1 + ep ** 2)) ** 2


def ladder_s21_s11(gk, g_load, omega_norm):
    """
    ABCD matrix method: compute S21 and S11 at normalized frequency.
    gk: prototype g-values [g1, g2, ..., gN]
    omega_norm: normalized angular frequency (1 = cutoff)
    Returns: (|S21|^2, |S11|^2)
    """
    j = 1j
    s = j * omega_norm
    A, B, C, D = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j

    for k, g in enumerate(gk):
        if k % 2 == 0:  # series inductor
            Z = s * g
            A_new = A + C * Z
            B_new = B + D * Z
            A, B = A_new, B_new
        else:  # shunt capacitor
            Y = s * g
            C_new = C + A * Y
            D_new = D + B * Y
            C, D = C_new, D_new

    R_L = g_load
    denom = A * R_L + B + C * R_L + D
    s21 = 2.0 * np.sqrt(R_L) / denom
    s21_sq = abs(s21) ** 2

    # S11 = (A*R_L + B - C*R_L - D) / (A*R_L + B + C*R_L + D)
    s11 = (A * R_L + B - C * R_L - D) / denom
    s11_sq = abs(s11) ** 2

    return s21_sq, s11_sq


def compute_lpf_sparam(ripple_db, N, fc, fs, num_points=500):
    """
    Compute S21 and S11 vs frequency for a Chebyshev LPF.
    Returns: freqs_hz, S21_dB, S11_dB
    """
    gk = compute_chebyshev_gk(ripple_db, N)
    g_load = compute_g_load(ripple_db, N)

    # Sweep from 0.01*fc to 3*fs
    f_max = max(3.0 * fs, 4.0 * fc)
    freqs_hz = np.linspace(0.01 * fc, f_max, num_points)
    omega_norm = freqs_hz / fc  # normalized to cutoff

    S21_dB = []
    S11_dB = []
    for w in omega_norm:
        s21_sq, s11_sq = ladder_s21_s11(gk, g_load, w)
        S21_dB.append(10 * np.log10(max(s21_sq, 1e-20)))
        S11_dB.append(10 * np.log10(max(s11_sq, 1e-20)))

    return freqs_hz, np.array(S21_dB), np.array(S11_dB)


def compute_bpf_sparam(ripple_db, N, f0, bw, num_points=500):
    """
    Compute S21 and S11 for a Chebyshev BPF using LP-to-BP mapping.
    """
    gk = compute_chebyshev_gk(ripple_db, N)
    g_load = compute_g_load(ripple_db, N)
    delta = bw / f0  # fractional bandwidth

    f_min = max(0.1 * f0, f0 - 3 * bw)
    f_max = f0 + 3 * bw
    freqs_hz = np.linspace(f_min, f_max, num_points)

    S21_dB = []
    S11_dB = []
    for f in freqs_hz:
        # LP-to-BP frequency mapping: omega_lp = (1/delta) * (f/f0 - f0/f)
        omega_lp = abs((1.0 / delta) * (f / f0 - f0 / f))
        s21_sq, s11_sq = ladder_s21_s11(gk, g_load, omega_lp)
        S21_dB.append(10 * np.log10(max(s21_sq, 1e-20)))
        S11_dB.append(10 * np.log10(max(s11_sq, 1e-20)))

    return freqs_hz, np.array(S21_dB), np.array(S11_dB)


# ===== FIGURE 1: LPF S-parameter comparison (correct vs wrong order) =====

def plot_lpf_comparison():
    """LPF: fc=1GHz, fs=2GHz, ripple=0.1dB, La≥40dB
    Correct order=5, Wrong prediction=3
    """
    fc = 1e9
    fs = 2e9
    ripple = 0.1

    freqs_5, s21_5, s11_5 = compute_lpf_sparam(ripple, 5, fc, fs)
    freqs_3, s21_3, s11_3 = compute_lpf_sparam(ripple, 3, fc, fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # S21
    ax1.plot(freqs_5 / 1e9, s21_5, '-', color='#1565C0', linewidth=2, label='N=5 (模型预测, 正确)')
    ax1.plot(freqs_3 / 1e9, s21_3, '--', color='#E53935', linewidth=2, label='N=3 (原始模型, 错误)')
    ax1.axhline(y=-40, color='#4CAF50', linestyle=':', linewidth=1.5, alpha=0.7, label='衰减目标 -40 dB')
    ax1.axvline(x=fc / 1e9, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=fs / 1e9, color='gray', linestyle=':', alpha=0.5)
    ax1.text(fc / 1e9, -5, '$f_c$', ha='center', fontsize=10, color='gray')
    ax1.text(fs / 1e9, -5, '$f_s$', ha='center', fontsize=10, color='gray')
    ax1.fill_betweenx([-80, 0], 0, fc / 1e9, alpha=0.05, color='#2196F3', label='通带')
    ax1.fill_betweenx([-80, 0], fs / 1e9, 6, alpha=0.05, color='#F44336', label='阻带')
    ax1.set_ylabel('$|S_{21}|$ (dB)', fontsize=12)
    ax1.set_ylim(-80, 5)
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Chebyshev 低通滤波器 S 参数对比\n($f_c$=1 GHz, $f_s$=2 GHz, Ripple=0.1 dB, $L_A$≥40 dB)',
                   fontsize=13, fontweight='bold')

    # S11
    ax2.plot(freqs_5 / 1e9, s11_5, '-', color='#1565C0', linewidth=2, label='N=5 (模型预测, 正确)')
    ax2.plot(freqs_3 / 1e9, s11_3, '--', color='#E53935', linewidth=2, label='N=3 (原始模型, 错误)')
    ax2.axhline(y=-10, color='#FF9800', linestyle=':', linewidth=1.5, alpha=0.7, label='S11 < -10 dB')
    ax2.axvline(x=fc / 1e9, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('$|S_{11}|$ (dB)', fontsize=12)
    ax2.set_xlabel('频率 (GHz)', fontsize=12)
    ax2.set_ylim(-50, 5)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'sparam_lpf_comparison.png')
    fig.savefig(SAVE_DIR / 'sparam_lpf_comparison.pdf')
    plt.close(fig)
    print('[OK] LPF S参数对比 -> sparam_lpf_comparison.pdf')


# ===== FIGURE 2: BPF S-parameter comparison =====

def plot_bpf_comparison():
    """BPF: f0=1.5GHz, BW=200MHz, ripple=0.1dB
    Correct order=7, Wrong=4
    """
    f0 = 1.5e9
    bw = 0.2e9
    ripple = 0.1

    freqs_7, s21_7, s11_7 = compute_bpf_sparam(ripple, 7, f0, bw)
    freqs_4, s21_4, s11_4 = compute_bpf_sparam(ripple, 4, f0, bw)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(freqs_7 / 1e9, s21_7, '-', color='#1565C0', linewidth=2, label='N=7 (微调模型, 正确)')
    ax1.plot(freqs_4 / 1e9, s21_4, '--', color='#E53935', linewidth=2, label='N=4 (原始模型, 错误)')
    ax1.axhline(y=-30, color='#4CAF50', linestyle=':', linewidth=1.5, alpha=0.7, label='衰减目标 -30 dB')
    ax1.axvline(x=(f0 - bw / 2) / 1e9, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=(f0 + bw / 2) / 1e9, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=f0 / 1e9, color='#9C27B0', linestyle='--', alpha=0.3)
    ax1.text(f0 / 1e9, -3, '$f_0$', ha='center', fontsize=10, color='#9C27B0')
    ax1.set_ylabel('$|S_{21}|$ (dB)', fontsize=12)
    ax1.set_ylim(-70, 5)
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Chebyshev 带通滤波器 S 参数对比\n($f_0$=1.5 GHz, BW=200 MHz, Ripple=0.1 dB)',
                   fontsize=13, fontweight='bold')

    ax2.plot(freqs_7 / 1e9, s11_7, '-', color='#1565C0', linewidth=2, label='N=7 (微调模型, 正确)')
    ax2.plot(freqs_4 / 1e9, s11_4, '--', color='#E53935', linewidth=2, label='N=4 (原始模型, 错误)')
    ax2.axhline(y=-10, color='#FF9800', linestyle=':', linewidth=1.5, alpha=0.7, label='S11 < -10 dB')
    ax2.set_ylabel('$|S_{11}|$ (dB)', fontsize=12)
    ax2.set_xlabel('频率 (GHz)', fontsize=12)
    ax2.set_ylim(-50, 5)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'sparam_bpf_comparison.png')
    fig.savefig(SAVE_DIR / 'sparam_bpf_comparison.pdf')
    plt.close(fig)
    print('[OK] BPF S参数对比 -> sparam_bpf_comparison.pdf')


# ===== FIGURE 3: Order impact on stopband attenuation =====

def plot_order_impact():
    """Show how filter order affects S21 stopband attenuation for fixed specs."""
    fc = 1e9
    fs = 2e9
    ripple = 0.1

    orders = range(2, 10)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(list(orders))))

    for i, N in enumerate(orders):
        freqs, s21, _ = compute_lpf_sparam(ripple, N, fc, fs, num_points=400)
        ax.plot(freqs / 1e9, s21, linewidth=1.8, color=colors[i], label=f'N={N}', alpha=0.85)

    # Highlight correct order
    ax.axhline(y=-40, color='#E53935', linestyle='--', linewidth=1.5, alpha=0.7, label='$L_A$=40 dB 目标')
    ax.axvline(x=fc / 1e9, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=fs / 1e9, color='gray', linestyle=':', alpha=0.4)
    ax.text(0.5, -3, '通带', ha='center', fontsize=10, color='#1565C0')
    ax.text(3.5, -3, '阻带', ha='center', fontsize=10, color='#E53935')

    ax.set_xlabel('频率 (GHz)', fontsize=12)
    ax.set_ylabel('$|S_{21}|$ (dB)', fontsize=12)
    ax.set_title('滤波器阶数对 $|S_{21}|$ 频率响应的影响\n($f_c$=1 GHz, $f_s$=2 GHz, Chebyshev, Ripple=0.1 dB)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-100, 5)
    ax.set_xlim(0, 6)
    ax.legend(fontsize=9, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotation: N=5 is the minimum order meeting La>=40dB
    ax.annotate('N≥5 满足衰减要求',
                xy=(2.0, -42), xytext=(3.5, -25),
                fontsize=10, color='#E53935', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'order_impact_sparam.png')
    fig.savefig(SAVE_DIR / 'order_impact_sparam.pdf')
    plt.close(fig)
    print('[OK] 阶数影响图 -> order_impact_sparam.pdf')


if __name__ == '__main__':
    print('生成S参数图表...')
    plot_lpf_comparison()
    plot_bpf_comparison()
    plot_order_impact()
    print('全部完成!')
