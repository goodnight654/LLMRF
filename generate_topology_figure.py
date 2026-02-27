#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Type-I 梯形滤波器拓扑结构示意图 (N=5 为例)
用于指导 Visio 绘制

拓扑:
  Rs ─┤ L1 ├─┬─┤ L3 ├─┬─┤ L5 ├─┬─ RL
              │        │        │
              C2       C4       (end)
              │        │
             GND      GND

Type-I: g1=series L, g2=shunt C, g3=series L, ...
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
C_WIRE = '#333333'
C_RS = '#607D8B'
C_RL = '#607D8B'
C_L = '#1976D2'    # Blue for inductors
C_C = '#E64A19'    # Orange for capacitors
C_GND = '#555555'
C_TEXT = '#111111'
LW = 2.0

def draw_inductor(ax, x_start, x_end, y, label='', color=C_L):
    """Draw inductor symbol (coil)"""
    xm = (x_start + x_end) / 2
    w = x_end - x_start
    # Box
    rect = patches.FancyBboxPatch((x_start + 0.05, y - 0.25), w - 0.1, 0.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E3F2FD', edgecolor=color, linewidth=LW)
    ax.add_patch(rect)
    ax.text(xm, y, label, ha='center', va='center', fontsize=10,
            fontweight='bold', color=color, fontfamily='serif')

def draw_capacitor(ax, x, y_top, y_bot, label='', color=C_C):
    """Draw capacitor symbol (two plates)"""
    gap = 0.12
    plate_w = 0.4
    ym = (y_top + y_bot) / 2
    # top wire
    ax.plot([x, x], [y_top, ym + gap], color=C_WIRE, linewidth=LW)
    # bottom wire
    ax.plot([x, x], [ym - gap, y_bot], color=C_WIRE, linewidth=LW)
    # plates
    ax.plot([x - plate_w/2, x + plate_w/2], [ym + gap, ym + gap],
            color=color, linewidth=LW + 1)
    ax.plot([x - plate_w/2, x + plate_w/2], [ym - gap, ym - gap],
            color=color, linewidth=LW + 1)
    ax.text(x + 0.35, ym, label, ha='left', va='center', fontsize=9,
            color=color, fontfamily='serif')

def draw_gnd(ax, x, y):
    """Draw ground symbol"""
    w = 0.3
    ax.plot([x, x], [y, y - 0.15], color=C_GND, linewidth=LW)
    for i, factor in enumerate([1.0, 0.65, 0.3]):
        yy = y - 0.15 - i * 0.08
        ax.plot([x - w/2 * factor, x + w/2 * factor], [yy, yy],
                color=C_GND, linewidth=1.5)

def draw_resistor(ax, x_start, x_end, y, label='', color=C_RS):
    """Draw resistor box"""
    xm = (x_start + x_end) / 2
    w = x_end - x_start
    rect = patches.FancyBboxPatch((x_start + 0.05, y - 0.2), w - 0.1, 0.4,
                                   boxstyle="round,pad=0.03",
                                   facecolor='#ECEFF1', edgecolor=color, linewidth=LW)
    ax.add_patch(rect)
    ax.text(xm, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=color, fontfamily='serif')

# Main signal line y=1.0
y_main = 1.0
y_gnd = -1.2

# Source
ax.plot([0, 0.8], [y_main, y_main], color=C_WIRE, linewidth=LW)
ax.text(0.0, y_main + 0.35, 'Port 1', ha='center', fontsize=8, color=C_TEXT)
# Dot at source
ax.plot(0, y_main, 'o', color=C_WIRE, markersize=5)

# Rs
draw_resistor(ax, 0.8, 1.8, y_main, '$R_s$', C_RS)
ax.text(1.3, y_main - 0.5, '$g_0 = 1$', ha='center', fontsize=7, color='gray', style='italic')

# Wire to L1
ax.plot([1.8, 2.2], [y_main, y_main], color=C_WIRE, linewidth=LW)

# L1 (series inductor)
draw_inductor(ax, 2.2, 3.5, y_main, '$L_1$')
ax.text(2.85, y_main + 0.55, '$g_1$', ha='center', fontsize=8, color='gray', style='italic')

# Wire L1 to node
ax.plot([3.5, 4.2], [y_main, y_main], color=C_WIRE, linewidth=LW)
# Node dot
ax.plot(4.2, y_main, 'o', color=C_WIRE, markersize=4)

# C2 (shunt capacitor)
draw_capacitor(ax, 4.2, y_main, y_gnd + 0.4, '$C_2$')
ax.text(4.2, y_main + 0.35, '$g_2$', ha='center', fontsize=8, color='gray', style='italic')
draw_gnd(ax, 4.2, y_gnd + 0.4)

# Wire to L3
ax.plot([4.2, 4.8], [y_main, y_main], color=C_WIRE, linewidth=LW)

# L3 (series inductor)
draw_inductor(ax, 4.8, 6.1, y_main, '$L_3$')
ax.text(5.45, y_main + 0.55, '$g_3$', ha='center', fontsize=8, color='gray', style='italic')

# Wire L3 to node
ax.plot([6.1, 6.8], [y_main, y_main], color=C_WIRE, linewidth=LW)
# Node dot
ax.plot(6.8, y_main, 'o', color=C_WIRE, markersize=4)

# C4 (shunt capacitor)
draw_capacitor(ax, 6.8, y_main, y_gnd + 0.4, '$C_4$')
ax.text(6.8, y_main + 0.35, '$g_4$', ha='center', fontsize=8, color='gray', style='italic')
draw_gnd(ax, 6.8, y_gnd + 0.4)

# Wire to L5
ax.plot([6.8, 7.4], [y_main, y_main], color=C_WIRE, linewidth=LW)

# L5 (series inductor)
draw_inductor(ax, 7.4, 8.7, y_main, '$L_5$')
ax.text(8.05, y_main + 0.55, '$g_5$', ha='center', fontsize=8, color='gray', style='italic')

# Wire to RL
ax.plot([8.7, 9.3], [y_main, y_main], color=C_WIRE, linewidth=LW)

# RL
draw_resistor(ax, 9.3, 10.3, y_main, '$R_L$', C_RL)
ax.text(9.8, y_main - 0.5, '$g_6$', ha='center', fontsize=7, color='gray', style='italic')

# Wire to port 2
ax.plot([10.3, 11.0], [y_main, y_main], color=C_WIRE, linewidth=LW)
ax.text(11.0, y_main + 0.35, 'Port 2', ha='center', fontsize=8, color=C_TEXT)
ax.plot(11.0, y_main, 'o', color=C_WIRE, markersize=5)

# Ground return line
ax.plot([0, 0], [y_main, y_gnd], color=C_WIRE, linewidth=LW)
ax.plot([0, 11.0], [y_gnd, y_gnd], color=C_WIRE, linewidth=LW)
ax.plot([11.0, 11.0], [y_gnd, y_main], color=C_WIRE, linewidth=LW)
draw_gnd(ax, 5.5, y_gnd - 0.05)

# Title annotation
ax.text(5.5, 2.2, 'Chebyshev Type-I Ladder Topology ($N=5$)',
        ha='center', fontsize=12, fontweight='bold', color=C_TEXT, fontfamily='serif')
ax.text(5.5, 1.8, 'Series inductor first: $g_1{=}L$, $g_2{=}C$, $g_3{=}L$, $g_4{=}C$, $g_5{=}L$',
        ha='center', fontsize=8, color='gray', fontfamily='serif')

fig.tight_layout()

save_dir = Path(__file__).parent / 'paper_materials' / 'figures'
save_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(save_dir / 'type1_topology.pdf', bbox_inches='tight')
fig.savefig(save_dir / 'type1_topology.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_dir / 'type1_topology.pdf'}")
print(f"Saved: {save_dir / 'type1_topology.png'}")
