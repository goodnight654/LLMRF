"""
生成三方消融实验柱状图
Qwen3-8B (No FT) vs QLoRA-Early (~11k) vs QLoRA-Full (Ours, ~25k)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ── 数据 ──
models = ['Qwen3-8B\n(No Fine-tuning)', 'QLoRA-Early\n(~11k samples)', 'QLoRA-Full\n(Ours, ~25k)']
colors = ['#90CAF9', '#FFB74D', '#66BB6A']
edge_colors = ['#1565C0', '#E65100', '#2E7D32']

# ── Fig A: Overall order accuracy ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Overall order accuracy
overall = [19.5, 12.6, 83.4]
bars = axes[0].bar(models, overall, color=colors, edgecolor=edge_colors, linewidth=1.5, width=0.55)
for bar, val in zip(bars, overall):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 画箭头标注
axes[0].annotate('', xy=(2, 83.4), xytext=(0, 19.5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
axes[0].text(1.2, 55, '+63.9%', fontsize=13, color='red', fontweight='bold', ha='center')

axes[0].annotate('', xy=(1, 12.6), xytext=(0, 19.5),
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5, ls='--'))
axes[0].text(0.6, 10, '−6.9%↓', fontsize=10, color='#B71C1C', fontweight='bold', ha='center')

axes[0].set_ylabel('Order Accuracy (%)', fontsize=12)
axes[0].set_title('(a) Overall Order Accuracy', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].axhline(y=50, color='gray', linestyle=':', alpha=0.5)
axes[0].grid(axis='y', alpha=0.3)

# Right: By-band order accuracy
bands = ['LPF', 'HPF', 'BPF']
no_ft   = [26.3, 23.3, 0.0]
early   = [7.9, 10.0, 26.3]
full    = [98.6, 98.2, 47.1]

x = np.arange(len(bands))
w = 0.25
b1 = axes[1].bar(x - w, no_ft, w, label='Qwen3-8B (No FT)', color=colors[0], edgecolor=edge_colors[0], linewidth=1.2)
b2 = axes[1].bar(x, early, w, label='QLoRA-Early (~11k)', color=colors[1], edgecolor=edge_colors[1], linewidth=1.2)
b3 = axes[1].bar(x + w, full, w, label='QLoRA-Full (Ours)', color=colors[2], edgecolor=edge_colors[2], linewidth=1.2)

for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        h = bar.get_height()
        if h > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_xticks(x)
axes[1].set_xticklabels(bands, fontsize=12)
axes[1].set_ylabel('Order Accuracy (%)', fontsize=12)
axes[1].set_title('(b) Order Accuracy by Filter Band', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 115)
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Three-Way Ablation Study: Impact of Data Quality & Augmentation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

for ext in ['pdf', 'png']:
    plt.savefig(f'paper_materials/figures/ablation_three_way.{ext}')
    plt.savefig(f'paper_materials/latex/figs/ablation_three_way.{ext}')
print("✓ ablation_three_way.pdf/png generated")
plt.close()
