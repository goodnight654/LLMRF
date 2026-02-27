#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF-FilterLLM Architecture Figure Generator (v2 - polished)
Top-conference-quality system architecture diagram.
Style: flat vector, minimalist, DeepMind/OpenAI aesthetic.
Output: paper_materials/figures/architecture.pdf & .png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
from pathlib import Path

SAVE_DIR = Path(__file__).parent / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════ COLOR PALETTE ═══════════════════════
# Soft pastels — professional academic aesthetic

# Module A - Data Engine (blue)
A_BG     = '#EDF4FC';  A_BORDER = '#5B9BD5';  A_HEADER = '#4179A8'
A_BOX    = '#D4E6F6';  A_TEXT   = '#2D5F8A'

# Module B - Fine-Tuning (green)
B_BG     = '#EEF7EF';  B_BORDER = '#6AAF7B';  B_HEADER = '#4D9260'
B_BOX    = '#D2EDCF';  B_TEXT   = '#3A7048'

# Module C - Closed-Loop (warm amber)
C_BG     = '#FFF6ED';  C_BORDER = '#D4903E';  C_HEADER = '#BD7D2E'
C_BOX    = '#FAE8CA';  C_TEXT   = '#7D5520'

# Augmentation sub-colors
AUG_PIR  = '#EAD8F2';  AUG_PIR_B = '#A87CC0'
AUG_BPP  = '#D5EBF2';  AUG_BPP_B = '#5A9DB0'
AUG_CUR  = '#F2EBD5';  AUG_CUR_B = '#C0A040'

# Decision & status
PASS_BG  = '#D4EDDA';  PASS_BD  = '#5CB85C'
FAIL_BG  = '#F8D7DA';  FAIL_BD  = '#D9534F'
ARROW    = '#5A5E65'
LABEL    = '#2D2D2D'
GRAY     = '#888888'
LGRAY    = '#AAAAAA'

# ═══════════════════════ DRAWING HELPERS ═══════════════════════

def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=0.12, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)

def cylinder(ax, x, y, w, h, fc, ec, lw=1.2, z=3):
    """Simplified database cylinder."""
    body = FancyBboxPatch((x, y), w, h*0.75,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(body)
    ell = mpatches.Ellipse((x+w/2, y+h*0.75), w, h*0.38,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z+1)
    ax.add_patch(ell)

def arrow(ax, x1, y1, x2, y2, c=ARROW, lw=1.5, style='->', cs='arc3,rad=0',
          z=5, ls='-'):
    a = FancyArrowPatch((x1,y1),(x2,y2), connectionstyle=cs,
            arrowstyle=style, mutation_scale=13, linewidth=lw,
            color=c, zorder=z, linestyle=ls)
    ax.add_patch(a)

def diamond(ax, cx, cy, w, h, fc, ec, lw=1.2, z=3):
    vs = [(cx, cy+h/2),(cx+w/2, cy),(cx, cy-h/2),(cx-w/2, cy),(cx, cy+h/2)]
    cs = [MplPath.MOVETO]+[MplPath.LINETO]*3+[MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(vs, cs),
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def T(ax, x, y, s, fs=7, c=LABEL, w='normal', ha='center', va='center',
      z=10, sty='normal'):
    ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va,
            zorder=z, style=sty, fontfamily='sans-serif')

# ═══════════════════════ MAIN FIGURE ═══════════════════════

def generate_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(17.5, 8.8))
    ax.set_xlim(-0.3, 18.2)
    ax.set_ylim(-1.2, 8.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ─────────── MODULE A : Data Engine ───────────
    Ax, Ay, Aw, Ah = 0, 0.3, 5.4, 7.5
    rbox(ax, Ax, Ay, Aw, Ah, A_BG, A_BORDER, lw=1.8, r=0.18, z=1)
    rbox(ax, Ax+0.12, Ay+Ah-0.62, Aw-0.24, 0.5, A_HEADER, A_HEADER, r=0.1, z=3)
    T(ax, Ax+Aw/2, Ay+Ah-0.37, 'Stage I : Data Engine', fs=9.5, c='white', w='bold')

    # sub-box dimensions
    sw, sh = 4.7, 0.62
    sx = Ax + (Aw-sw)/2

    # 1) Parameter Sampling
    y1 = 6.2
    rbox(ax, sx, y1, sw, sh, A_BOX, A_BORDER, r=0.09)
    T(ax, sx+sw/2, y1+sh/2+0.1, 'Parameter Space Sampling', fs=7.2, c=A_TEXT, w='bold')
    T(ax, sx+sw/2, y1+sh/2-0.14, r'$f_c, f_s, L_r, L_A, R_0$  ×  LPF / HPF / BPF', fs=5.5, c=GRAY, sty='italic')
    arrow(ax, Ax+Aw/2, y1, Ax+Aw/2, y1-0.32)

    # 2) Order Computation
    y2 = 5.2
    rbox(ax, sx, y2, sw, sh, A_BOX, A_BORDER, r=0.09)
    T(ax, sx+sw/2, y2+sh/2+0.1, 'Chebyshev Order Computation', fs=7.2, c=A_TEXT, w='bold')
    T(ax, sx+sw/2, y2+sh/2-0.14, r'$N = \lceil \cosh^{-1}(\cdots)/\cosh^{-1}(k_s) \rceil$', fs=5.8, c=GRAY)
    arrow(ax, Ax+Aw/2, y2, Ax+Aw/2, y2-0.32)

    # 3) Prototype Synthesis + Multi-band
    y3 = 4.2
    rbox(ax, sx, y3, sw, sh, A_BOX, A_BORDER, r=0.09)
    T(ax, sx+sw/2, y3+sh/2+0.1, 'Prototype Synthesis & Freq. Transform', fs=7, c=A_TEXT, w='bold')
    T(ax, sx+sw/2, y3+sh/2-0.14, r'$g_k$ → L, C elements  |  LPF · HPF · BPF', fs=5.5, c=GRAY, sty='italic')

    # Arrows to 3 augmentation boxes
    aug_top = 3.7
    arrow(ax, Ax+Aw/2 - 1.0, y3, sx + sw*0.17, aug_top, cs='arc3,rad=0.12')
    arrow(ax, Ax+Aw/2,       y3, sx + sw*0.50, aug_top)
    arrow(ax, Ax+Aw/2 + 1.0, y3, sx + sw*0.83, aug_top, cs='arc3,rad=-0.12')

    # 4) Three Augmentation Pipelines
    aw_ = 1.42;  ah_ = 1.05;  ag = 0.12
    ay_ = 2.5

    # PIR
    px = sx + 0.02
    rbox(ax, px, ay_, aw_, ah_, AUG_PIR, AUG_PIR_B, lw=1.0, r=0.09)
    T(ax, px+aw_/2, ay_+ah_-0.18, 'PIR', fs=7, c='#6A3D7D', w='bold')
    T(ax, px+aw_/2, ay_+ah_/2-0.02, 'Perturbation', fs=5.2, c='#6A3D7D')
    T(ax, px+aw_/2, ay_+ah_/2-0.2, 'Reflection', fs=5.2, c='#6A3D7D')
    T(ax, px+aw_/2, ay_+0.1, 'fail → diagnose → fix', fs=4.5, c=LGRAY, sty='italic')

    # BPP
    bx = px + aw_ + ag
    rbox(ax, bx, ay_, aw_, ah_, AUG_BPP, AUG_BPP_B, lw=1.0, r=0.09)
    T(ax, bx+aw_/2, ay_+ah_-0.18, 'BPP', fs=7, c='#3A7A8D', w='bold')
    T(ax, bx+aw_/2, ay_+ah_/2-0.02, 'Bidirectional', fs=5.2, c='#3A7A8D')
    T(ax, bx+aw_/2, ay_+ah_/2-0.2, 'Prediction', fs=5.2, c='#3A7A8D')
    T(ax, bx+aw_/2, ay_+0.1, 'spec ↔ param', fs=4.5, c=LGRAY, sty='italic')

    # Curriculum
    cx_ = bx + aw_ + ag
    rbox(ax, cx_, ay_, aw_, ah_, AUG_CUR, AUG_CUR_B, lw=1.0, r=0.09)
    T(ax, cx_+aw_/2, ay_+ah_-0.18, 'Curriculum', fs=7, c='#8B6914', w='bold')
    T(ax, cx_+aw_/2, ay_+ah_/2-0.02, 'Easy → Hard', fs=5.2, c='#8B6914')
    T(ax, cx_+aw_/2, ay_+ah_/2-0.2, 'Ordering', fs=5.2, c='#8B6914')
    T(ax, cx_+aw_/2, ay_+0.1, 'difficulty score', fs=4.5, c=LGRAY, sty='italic')

    # Arrows to dataset
    cyl_top = 2.15
    arrow(ax, px+aw_/2, ay_, sx+sw*0.25, cyl_top, cs='arc3,rad=0.08')
    arrow(ax, bx+aw_/2, ay_, sx+sw*0.50, cyl_top)
    arrow(ax, cx_+aw_/2, ay_, sx+sw*0.75, cyl_top, cs='arc3,rad=-0.08')

    # 5) Dataset cylinder
    dc_w, dc_h = 3.2, 0.85
    dc_x = Ax + (Aw - dc_w)/2
    dc_y = 0.65
    cylinder(ax, dc_x, dc_y, dc_w, dc_h, '#C0D9F0', A_BORDER)
    T(ax, dc_x+dc_w/2, dc_y+dc_h*0.42, 'RF-Filter-SFT', fs=7.5, c=A_TEXT, w='bold')
    T(ax, dc_x+dc_w/2, dc_y+dc_h*0.08, '25,168 train  |  3 bands  |  ZH+EN', fs=5, c=GRAY)

    # ─────────── MODULE B : Fine-Tuning ───────────
    Bx, By, Bw, Bh = 6.1, 0.3, 3.8, 7.5
    rbox(ax, Bx, By, Bw, Bh, B_BG, B_BORDER, lw=1.8, r=0.18, z=1)
    rbox(ax, Bx+0.12, By+Bh-0.62, Bw-0.24, 0.5, B_HEADER, B_HEADER, r=0.1, z=3)
    T(ax, Bx+Bw/2, By+Bh-0.37, 'Stage II : Fine-Tuning', fs=9.5, c='white', w='bold')

    bw_, bh_ = 3.1, 0.62
    bsx = Bx + (Bw - bw_)/2

    # Base model
    by1 = 6.2
    rbox(ax, bsx, by1, bw_, bh_, B_BOX, B_BORDER, r=0.09)
    T(ax, bsx+bw_/2, by1+bh_/2+0.1, 'LLM Backbone', fs=7.2, c=B_TEXT, w='bold')
    T(ax, bsx+bw_/2, by1+bh_/2-0.14, '8B params · 36 layers', fs=5.5, c=GRAY)
    arrow(ax, Bx+Bw/2, by1, Bx+Bw/2, by1-0.35)

    # NF4
    by2 = 5.2
    rbox(ax, bsx, by2, bw_, 0.5, B_BOX, B_BORDER, r=0.09)
    T(ax, bsx+bw_/2, by2+0.25, 'NF4 Quantization (4-bit)', fs=7, c=B_TEXT, w='bold')
    arrow(ax, Bx+Bw/2, by2, Bx+Bw/2, by2-0.35)

    # LoRA
    by3 = 4.15
    rbox(ax, bsx, by3, bw_, 0.7, B_BOX, B_BORDER, r=0.09, lw=1.5)
    T(ax, bsx+bw_/2, by3+0.5, 'LoRA Adapter', fs=7.2, c=B_TEXT, w='bold')
    T(ax, bsx+bw_/2, by3+0.27, 'r = 8,  α = 16', fs=5.5, c=GRAY)
    T(ax, bsx+bw_/2, by3+0.08, '0.047% trainable', fs=5.5, c='#C0392B', w='bold')
    arrow(ax, Bx+Bw/2, by3, Bx+Bw/2, by3-0.35)

    # Training
    by4 = 3.05
    rbox(ax, bsx, by4, bw_, 0.7, B_BOX, B_BORDER, r=0.09)
    T(ax, bsx+bw_/2, by4+0.5, 'QLoRA Training', fs=7.2, c=B_TEXT, w='bold')
    T(ax, bsx+bw_/2, by4+0.27, '1× RTX 4090  ·  9.4 h', fs=5.5, c=GRAY)
    T(ax, bsx+bw_/2, by4+0.08, 'LLaMA-Factory', fs=5.2, c=LGRAY, sty='italic')
    arrow(ax, Bx+Bw/2, by4, Bx+Bw/2, by4-0.55)

    # Metrics snapshot (small sub-box)
    by5 = 1.85
    rbox(ax, bsx, by5, bw_, 0.55, '#E8F5E9', B_BORDER, r=0.09, lw=0.8, alpha=0.7)
    T(ax, bsx+bw_/2, by5+0.35, 'Acc_N : 19.5% → 83.4%', fs=6, c='#2E7D32', w='bold')
    T(ax, bsx+bw_/2, by5+0.12, '100% JSON  ·  100% type/band', fs=5, c=GRAY)
    arrow(ax, Bx+Bw/2, by5, Bx+Bw/2, by5-0.5)

    # Output cylinder
    oc_w, oc_h = 2.8, 0.8
    oc_x = Bx + (Bw - oc_w)/2
    oc_y = 0.55
    cylinder(ax, oc_x, oc_y, oc_w, oc_h, '#C8E6C9', B_BORDER)
    T(ax, oc_x+oc_w/2, oc_y+oc_h*0.42, 'RF-FilterLLM', fs=7.5, c=B_TEXT, w='bold')
    T(ax, oc_x+oc_w/2, oc_y+oc_h*0.08, 'Merged  ·  15.6 GB  ·  BF16', fs=5, c=GRAY)

    # ─────────── MODULE C : Closed-Loop Agent ───────────
    Cx, Cy, Cw, Ch = 10.6, 0.3, 7.2, 7.5
    rbox(ax, Cx, Cy, Cw, Ch, C_BG, C_BORDER, lw=1.8, r=0.18, z=1)
    rbox(ax, Cx+0.12, Cy+Ch-0.62, Cw-0.24, 0.5, C_HEADER, C_HEADER, r=0.1, z=3)
    T(ax, Cx+Cw/2, Cy+Ch-0.37, 'Stage III : Closed-Loop Agent', fs=9.5, c='white', w='bold')

    # Central column
    fw = 3.0;  fh = 0.58
    fx = Cx + 1.9

    # User Spec
    uy = 6.2
    rbox(ax, fx, uy, fw, fh, '#F2E8D5', C_BORDER, r=0.09)
    T(ax, fx+fw/2, uy+fh/2+0.08, 'User Specification', fs=7.2, c=C_TEXT, w='bold')
    T(ax, fx+fw/2, uy+fh/2-0.14, 'natural-language input', fs=5.2, c=GRAY, sty='italic')
    arrow(ax, fx+fw/2, uy, fx+fw/2, uy-0.3)

    # LLM Inference
    ly = 5.25
    rbox(ax, fx, ly, fw, 0.65, C_BOX, C_BORDER, r=0.09, lw=1.6)
    T(ax, fx+fw/2, ly+0.45, 'LLM Inference', fs=7.5, c=C_TEXT, w='bold')
    T(ax, fx+fw/2, ly+0.22, 'RF-FilterLLM → JSON', fs=5.5, c=GRAY)
    T(ax, fx+fw/2, ly+0.05, '{N, L₁, C₁, L₂, ...}', fs=5, c=LGRAY, sty='italic')
    arrow(ax, fx+fw/2, ly, fx+fw/2, ly-0.3)

    # ADS Simulation
    sy = 4.3
    rbox(ax, fx, sy, fw, fh, C_BOX, C_BORDER, r=0.09)
    T(ax, fx+fw/2, sy+fh/2+0.08, 'EDA Simulation', fs=7.2, c=C_TEXT, w='bold')
    T(ax, fx+fw/2, sy+fh/2-0.14, 'Keysight ADS → S₁₁, S₂₁', fs=5.5, c=GRAY)
    arrow(ax, fx+fw/2, sy, fx+fw/2, sy-0.35)

    # Decision diamond
    dcx = fx + fw/2
    dcy = 3.5
    dw, dh = 1.3, 0.7
    diamond(ax, dcx, dcy, dw, dh, '#FFF8E1', C_BORDER, lw=1.2)
    T(ax, dcx, dcy+0.07, 'Evaluate', fs=6.5, c=C_TEXT, w='bold')
    T(ax, dcx, dcy-0.13, 'Pass?', fs=5.5, c=GRAY)

    # PASS (right)
    pw = 1.6;  ph = 0.48
    ppx = fx + fw + 0.65
    ppy = dcy - ph/2
    rbox(ax, ppx, ppy, pw, ph, PASS_BG, PASS_BD, r=0.09, lw=1.2)
    T(ax, ppx+pw/2, ppy+ph/2+0.06, '✓ PASS', fs=7, c='#2E7D32', w='bold')
    T(ax, ppx+pw/2, ppy+ph/2-0.14, 'Return Design', fs=5, c=GRAY)
    arrow(ax, dcx+dw/2, dcy, ppx, dcy, c=PASS_BD, lw=1.3)
    T(ax, dcx+dw/2+0.32, dcy+0.16, 'Yes', fs=5.5, c=PASS_BD, w='bold')

    # FAIL (down)
    fail_w = 1.1;  fail_h = 0.38
    fail_x = dcx - fail_w/2
    fail_y = 2.65
    rbox(ax, fail_x, fail_y, fail_w, fail_h, FAIL_BG, FAIL_BD, r=0.08, lw=1.0)
    T(ax, dcx, fail_y+fail_h/2, '✗ FAIL', fs=6.5, c='#C62828', w='bold')
    arrow(ax, dcx, dcy-dh/2, dcx, fail_y+fail_h, c=FAIL_BD, lw=1.3)
    T(ax, dcx+0.22, dcy-dh/2-0.12, 'No', fs=5.5, c=FAIL_BD, w='bold')

    # Arrow from FAIL to Sensitivity
    arrow(ax, dcx, fail_y, dcx, fail_y-0.3, c=FAIL_BD, lw=1.0, ls='--')

    # Sensitivity Analysis
    sa_y = 1.5
    sa_w = 3.0
    sa_x = fx
    rbox(ax, sa_x, sa_y, sa_w, 0.65, '#F5E6D0', C_BORDER, r=0.09, lw=1.3)
    T(ax, sa_x+sa_w/2, sa_y+0.46, 'ABCD Sensitivity Analysis', fs=7, c=C_TEXT, w='bold')
    T(ax, sa_x+sa_w/2, sa_y+0.22, 'Rank elements by impact', fs=5.5, c=GRAY)
    T(ax, sa_x+sa_w/2, sa_y+0.05, 'Top-k correction guidance', fs=5, c=LGRAY, sty='italic')

    # Feedback loop: Sensitivity → LLM Inference (curved left)
    ax.annotate('',
        xy=(fx, ly+0.32),
        xytext=(fx, sa_y+0.32),
        arrowprops=dict(
            arrowstyle='->', color=FAIL_BD,
            linewidth=2.0, linestyle='--',
            connectionstyle='arc3,rad=-0.55',
        ), zorder=5)
    # Label
    T(ax, fx-0.75, 3.5, 'Reflect', fs=5.5, c=FAIL_BD, w='bold')
    T(ax, fx-0.75, 3.22, '& Correct', fs=5.5, c=FAIL_BD, w='bold')
    T(ax, fx-0.75, 2.95, '≤ 5 iter', fs=5, c=FAIL_BD, sty='italic')

    # ═══════════════ INTER-MODULE ARROWS ═══════════════
    mid_y = 0.92
    # A → B
    arrow(ax, Ax+Aw, mid_y, Bx, mid_y, c=ARROW, lw=2.2, style='-|>')
    T(ax, (Ax+Aw+Bx)/2, mid_y+0.2, 'Training Data', fs=6, c=ARROW, w='bold')
    T(ax, (Ax+Aw+Bx)/2, mid_y-0.12, '25,168 samples', fs=5, c=LGRAY)

    # B → C
    arrow(ax, Bx+Bw, mid_y, Cx, mid_y, c=ARROW, lw=2.2, style='-|>')
    T(ax, (Bx+Bw+Cx)/2, mid_y+0.2, 'Model Deploy', fs=6, c=ARROW, w='bold')
    T(ax, (Bx+Bw+Cx)/2, mid_y-0.12, '15.6 GB BF16', fs=5, c=LGRAY)

    # ═══════════════ CONTRIBUTION BADGES ═══════════════
    by_ = -0.65
    badges = [
        ('C1: Dynamic Intent Completion',  A_BOX,    A_BORDER,   0.4),
        ('C2: Parameter-Efficient (0.047%)', B_BOX,  B_BORDER,   4.6),
        ('C3: Multi-Band  LPF / HPF / BPF', AUG_PIR, AUG_PIR_B, 8.8),
        ('C4: Sensitivity-Guided Reflection', C_BOX, C_BORDER,  13.0),
    ]
    for txt, bg, bd, bx_ in badges:
        bw_b = 3.9
        rbox(ax, bx_, by_, bw_b, 0.42, bg, bd, r=0.09, lw=0.9, alpha=0.85)
        T(ax, bx_+bw_b/2, by_+0.21, txt, fs=5.5, c=bd, w='bold')
    T(ax, 9.0, -0.2, 'Key Contributions', fs=6.5, c='#666666', w='bold')

    # ═══════════════ SAVE ═══════════════
    plt.tight_layout(pad=0.3)
    for fmt in ('pdf', 'png'):
        fig.savefig(SAVE_DIR / f'architecture.{fmt}', format=fmt,
                    bbox_inches='tight', pad_inches=0.12, facecolor='white',
                    dpi=300 if fmt == 'png' else 'figure')
    plt.close(fig)
    print("Done: architecture.pdf / .png saved to", SAVE_DIR)


if __name__ == '__main__':
    generate_architecture()
