#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文可视化图表生成脚本
Generate all figures for the paper on LLM-based RF filter design.

生成内容:
  1. 训练损失曲线 (Training Loss Curve)
  2. 基线 vs 微调 对比柱状图 (Baseline vs Fine-tuned Comparison)
  3. 分频段阶数准确率对比 (Order Accuracy by Band)
  4. 数据集分布饼图 (Dataset Distribution)
  5. 评估指标雷达图 (Evaluation Radar Chart)
  6. 训练学习率调度曲线 (Learning Rate Schedule)

输出目录: paper_materials/figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ============ 配置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11

SAVE_DIR = Path(__file__).parent / 'paper_materials' / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAINER_LOG = Path(__file__).parent / 'LLaMA-Factory' / 'saves' / 'Qwen3-8B-Base' / 'lora' / 'train_cleaned_v2' / 'trainer_log.jsonl'
BASELINE_COMP = Path(__file__).parent / 'paper_materials' / 'baseline_comparison.json'
EVAL_RESULTS = Path(__file__).parent / 'LLaMA-Factory' / 'saves' / 'Qwen3-8B-Base' / 'lora' / 'train_cleaned_v2' / 'eval_results_v2.json'
TRAIN_DATA = Path(__file__).parent / 'LLaMA-Factory' / 'data' / 'filter_sft_zhmix' / 'train.jsonl'


def load_trainer_log():
    """加载训练日志"""
    logs = []
    with open(TRAINER_LOG, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if 'loss' in entry:
                    logs.append(entry)
    return logs


def load_baseline_comparison():
    """加载基线对比数据"""
    with open(BASELINE_COMP, encoding='utf-8') as f:
        return json.load(f)


def load_eval_results():
    """加载评估结果"""
    with open(EVAL_RESULTS, encoding='utf-8') as f:
        return json.load(f)


def load_train_data():
    """加载训练数据统计"""
    data = []
    with open(TRAIN_DATA, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ============ 图1: 训练损失曲线 ============
def plot_training_loss(logs):
    """绘制训练损失曲线"""
    steps = [e['current_steps'] for e in logs]
    losses = [e['loss'] for e in logs]
    epochs = [e['epoch'] for e in logs]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss
    color1 = '#2196F3'
    ax1.plot(steps, losses, color=color1, linewidth=1.5, alpha=0.9)
    ax1.set_xlabel('训练步数 (Training Steps)', fontsize=12)
    ax1.set_ylabel('训练损失 (Training Loss)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 添加平滑曲线
    window = 15
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window//2:window//2+len(smoothed)]
        ax1.plot(smooth_steps, smoothed, color='#0D47A1', linewidth=2.5, label='平滑损失 (Smoothed)')

    # 标注关键点
    ax1.axhline(y=losses[-1], color='#E53935', linestyle='--', alpha=0.5, linewidth=1)
    ax1.annotate(f'最终损失: {losses[-1]:.4f}',
                 xy=(steps[-1], losses[-1]),
                 xytext=(steps[-1]*0.7, losses[-1]+0.15),
                 fontsize=10, color='#E53935',
                 arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))

    # Epoch 分界线
    epoch1_step = None
    for i, e in enumerate(epochs):
        if e >= 1.0 and epoch1_step is None:
            epoch1_step = steps[i]
    if epoch1_step:
        ax1.axvline(x=epoch1_step, color='#4CAF50', linestyle=':', alpha=0.7, linewidth=1.5)
        ax1.text(epoch1_step, max(losses)*0.95, ' Epoch 2', color='#4CAF50', fontsize=10,
                 verticalalignment='top')

    ax1.set_xlim(0, max(steps) * 1.02)
    ax1.set_ylim(0, max(losses) * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # 次坐标轴显示学习率
    ax2 = ax1.twinx()
    lrs = [e['lr'] for e in logs]
    color2 = '#FF9800'
    ax2.plot(steps, lrs, color=color2, linewidth=1, alpha=0.6, linestyle='--')
    ax2.set_ylabel('学习率 (Learning Rate)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.title('Qwen3-8B QLoRA 微调训练过程', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'training_loss_curve.png')
    fig.savefig(SAVE_DIR / 'training_loss_curve.pdf')
    plt.close(fig)
    print('[OK] 训练损失曲线 -> training_loss_curve.png/pdf')


# ============ 图2: 基线 vs 微调 对比柱状图 ============
def plot_baseline_comparison(comp_data):
    """绘制基线对比柱状图"""
    baseline = comp_data['baseline']
    finetuned = comp_data['finetuned']

    metrics = ['JSON解析\n成功率', '滤波器类型\n准确率', '滤波器频段\n准确率', '阶数\n准确率', '参数键\n完整率']
    b_total = baseline['total']
    f_total = finetuned['total']

    b_vals = [
        baseline['json_ok'] / b_total * 100,
        baseline['filter_type_ok'] / b_total * 100,
        baseline['filter_band_ok'] / b_total * 100,
        baseline['order_ok'] / b_total * 100,
        baseline['keys_complete'] / b_total * 100,
    ]
    f_vals = [
        finetuned['json_ok'] / f_total * 100,
        finetuned['filter_type_ok'] / f_total * 100,
        finetuned['filter_band_ok'] / f_total * 100,
        finetuned['order_ok'] / f_total * 100,
        finetuned['keys_complete'] / f_total * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, b_vals, width, label=f'Qwen3-8B (原始)', color='#90CAF9', edgecolor='#1565C0', linewidth=0.8)
    bars2 = ax.bar(x + width/2, f_vals, width, label=f'Qwen3-8B-RF (微调)', color='#EF9A9A', edgecolor='#B71C1C', linewidth=0.8)

    # 标注数值
    for bar, val in zip(bars1, b_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1565C0')
    for bar, val in zip(bars2, f_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B71C1C')

    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('原始模型 vs 微调模型 性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    # 高亮关键改进
    improvement = f_vals[3] - b_vals[3]
    ax.annotate(f'↑ {improvement:.1f}%',
                xy=(x[3] + width/2, f_vals[3]),
                xytext=(x[3] + 0.8, f_vals[3] + 10),
                fontsize=12, color='#D32F2F', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2))

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'baseline_comparison.png')
    fig.savefig(SAVE_DIR / 'baseline_comparison.pdf')
    plt.close(fig)
    print('[OK] 基线对比柱状图 -> baseline_comparison.png/pdf')


# ============ 图3: 分频段阶数准确率对比 ============
def plot_order_accuracy_by_band(comp_data):
    """绘制按频段分类的阶数准确率"""
    baseline = comp_data['baseline']
    finetuned = comp_data['finetuned']

    bands = ['lowpass', 'highpass', 'bandpass']
    band_labels = ['低通 (LPF)', '高通 (HPF)', '带通 (BPF)']

    b_acc = []
    f_acc = []
    b_counts = []
    f_counts = []

    for band in bands:
        bb = baseline['by_band'][band]
        fb = finetuned['by_band'][band]
        b_acc.append(bb['order_ok'] / bb['total'] * 100 if bb['total'] > 0 else 0)
        f_acc.append(fb['order_ok'] / fb['total'] * 100 if fb['total'] > 0 else 0)
        b_counts.append(bb['total'])
        f_counts.append(fb['total'])

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    colors_b = ['#BBDEFB', '#C8E6C9', '#FFE0B2']
    colors_f = ['#1565C0', '#2E7D32', '#E65100']

    bars1 = ax.bar(x - width/2, b_acc, width, label='Qwen3-8B (原始)',
                   color=['#90CAF9', '#A5D6A7', '#FFCC80'],
                   edgecolor=['#1565C0', '#2E7D32', '#E65100'], linewidth=1)
    bars2 = ax.bar(x + width/2, f_acc, width, label='Qwen3-8B-RF (微调)',
                   color=['#1565C0', '#2E7D32', '#E65100'],
                   edgecolor=['#0D47A1', '#1B5E20', '#BF360C'], linewidth=1)

    for bar, val, n in zip(bars1, b_acc, b_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)
    for bar, val, n in zip(bars2, f_acc, f_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('阶数准确率 (%)', fontsize=12)
    ax.set_title('各频段滤波器阶数预测准确率对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'order_accuracy_by_band.png')
    fig.savefig(SAVE_DIR / 'order_accuracy_by_band.pdf')
    plt.close(fig)
    print('[OK] 分频段阶数准确率 -> order_accuracy_by_band.png/pdf')


# ============ 图4: 数据集分布 ============
def plot_dataset_distribution(train_data):
    """绘制数据集样本类型和频段分布"""
    from collections import defaultdict

    by_type = defaultdict(int)
    by_band = defaultdict(int)
    by_lang = defaultdict(int)

    for s in train_data:
        msgs = s['messages']
        asst = [m for m in msgs if m['role'] == 'assistant']
        if not asst:
            continue
        first = asst[0]['content']
        try:
            j = json.loads(first)
            band = j.get('filter_band', 'unknown')
            by_type['full (直接设计)'] += 1
            by_band[band] += 1
        except:
            if len(asst) == 1:
                by_type['followup_question (追问)'] += 1
            else:
                by_type['followup_resolve (追问解答)'] += 1

        user = next(m['content'] for m in msgs if m['role'] == 'user')
        if any(c > '\u4e00' for c in user[:20]):
            by_lang['中文'] += 1
        else:
            by_lang['英文'] += 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 样本类型分布
    types = list(by_type.keys())
    type_vals = list(by_type.values())
    type_colors = ['#42A5F5', '#FFA726', '#66BB6A']
    wedges1, texts1, autotexts1 = axes[0].pie(type_vals, labels=types, autopct='%1.1f%%',
                                               colors=type_colors, startangle=90,
                                               textprops={'fontsize': 9})
    axes[0].set_title('样本类型分布', fontsize=12, fontweight='bold')

    # 频段分布
    band_map = {'lowpass': '低通 (LPF)', 'highpass': '高通 (HPF)', 'bandpass': '带通 (BPF)'}
    band_labels = [band_map.get(b, b) for b in by_band.keys()]
    band_vals = list(by_band.values())
    band_colors = ['#1565C0', '#2E7D32', '#E65100']
    wedges2, texts2, autotexts2 = axes[1].pie(band_vals, labels=band_labels, autopct='%1.1f%%',
                                               colors=band_colors, startangle=90,
                                               textprops={'fontsize': 10})
    for t in autotexts2:
        t.set_color('white')
        t.set_fontweight('bold')
    axes[1].set_title('滤波器频段分布', fontsize=12, fontweight='bold')

    # 语言分布
    lang_labels = list(by_lang.keys())
    lang_vals = list(by_lang.values())
    lang_colors = ['#EF5350', '#5C6BC0']
    wedges3, texts3, autotexts3 = axes[2].pie(lang_vals, labels=lang_labels, autopct='%1.1f%%',
                                               colors=lang_colors, startangle=90,
                                               textprops={'fontsize': 10})
    for t in autotexts3:
        t.set_fontweight('bold')
    axes[2].set_title('语言分布', fontsize=12, fontweight='bold')

    fig.suptitle(f'训练数据集分布 (总计 {len(train_data)} 样本)', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'dataset_distribution.png')
    fig.savefig(SAVE_DIR / 'dataset_distribution.pdf')
    plt.close(fig)
    print('[OK] 数据集分布 -> dataset_distribution.png/pdf')


# ============ 图5: 评估指标雷达图 ============
def plot_evaluation_radar(comp_data, eval_results):
    """绘制评估指标雷达图"""
    summary = eval_results['summary']

    # 基线和微调模型指标
    baseline = comp_data['baseline']
    finetuned = comp_data['finetuned']
    b_total = baseline['total']
    f_total = finetuned['total']

    categories = [
        'JSON解析', '滤波器类型', '滤波器频段',
        '阶数准确率', '参数完整性', 'LPF阶数',
        'HPF阶数', 'BPF阶数'
    ]

    # 使用基线对比中的数据
    b_vals = [
        baseline['json_ok'] / b_total * 100,
        baseline['filter_type_ok'] / b_total * 100,
        baseline['filter_band_ok'] / b_total * 100,
        baseline['order_ok'] / b_total * 100,
        baseline['keys_complete'] / b_total * 100,
        baseline['by_band']['lowpass']['order_ok'] / baseline['by_band']['lowpass']['total'] * 100,
        baseline['by_band']['highpass']['order_ok'] / baseline['by_band']['highpass']['total'] * 100,
        baseline['by_band']['bandpass']['order_ok'] / baseline['by_band']['bandpass']['total'] * 100,
    ]

    f_vals = [
        finetuned['json_ok'] / f_total * 100,
        finetuned['filter_type_ok'] / f_total * 100,
        finetuned['filter_band_ok'] / f_total * 100,
        finetuned['order_ok'] / f_total * 100,
        finetuned['keys_complete'] / f_total * 100,
        finetuned['by_band']['lowpass']['order_ok'] / finetuned['by_band']['lowpass']['total'] * 100,
        finetuned['by_band']['highpass']['order_ok'] / finetuned['by_band']['highpass']['total'] * 100,
        finetuned['by_band']['bandpass']['order_ok'] / finetuned['by_band']['bandpass']['total'] * 100,
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    b_vals += b_vals[:1]
    f_vals += f_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, b_vals, 'o-', linewidth=2, label='Qwen3-8B (原始)', color='#42A5F5', markersize=6)
    ax.fill(angles, b_vals, alpha=0.15, color='#42A5F5')
    ax.plot(angles, f_vals, 's-', linewidth=2, label='Qwen3-8B-RF (微调)', color='#EF5350', markersize=6)
    ax.fill(angles, f_vals, alpha=0.15, color='#EF5350')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_rlabel_position(30)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('模型评估指标雷达图', fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'evaluation_radar.png')
    fig.savefig(SAVE_DIR / 'evaluation_radar.pdf')
    plt.close(fig)
    print('[OK] 评估指标雷达图 -> evaluation_radar.png/pdf')


# ============ 图6: 综合结果表格图 ============
def plot_results_table(comp_data, eval_results):
    """绘制综合结果表格"""
    summary = eval_results['summary']
    baseline = comp_data['baseline']
    finetuned = comp_data['finetuned']
    b_total = baseline['total']
    f_total = finetuned['total']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # 表格数据
    col_labels = ['评估指标', 'Qwen3-8B\n(原始模型)', 'Qwen3-8B-RF\n(微调模型)', '提升幅度']
    row_data = [
        ['JSON解析成功率',
         f'{baseline["json_ok"]/b_total*100:.1f}%',
         f'{finetuned["json_ok"]/f_total*100:.1f}%',
         f'+{(finetuned["json_ok"]/f_total - baseline["json_ok"]/b_total)*100:.1f}%'],
        ['滤波器类型准确率',
         f'{baseline["filter_type_ok"]/b_total*100:.1f}%',
         f'{finetuned["filter_type_ok"]/f_total*100:.1f}%',
         f'+{(finetuned["filter_type_ok"]/f_total - baseline["filter_type_ok"]/b_total)*100:.1f}%'],
        ['滤波器频段准确率',
         f'{baseline["filter_band_ok"]/b_total*100:.1f}%',
         f'{finetuned["filter_band_ok"]/f_total*100:.1f}%',
         f'+{(finetuned["filter_band_ok"]/f_total - baseline["filter_band_ok"]/b_total)*100:.1f}%'],
        ['阶数准确率 (Overall)',
         f'{baseline["order_ok"]/b_total*100:.1f}%',
         f'{finetuned["order_ok"]/f_total*100:.1f}%',
         f'+{(finetuned["order_ok"]/f_total - baseline["order_ok"]/b_total)*100:.1f}%'],
        ['LPF 阶数准确率',
         f'{baseline["by_band"]["lowpass"]["order_ok"]/baseline["by_band"]["lowpass"]["total"]*100:.1f}%',
         f'{finetuned["by_band"]["lowpass"]["order_ok"]/finetuned["by_band"]["lowpass"]["total"]*100:.1f}%',
         f'+{(finetuned["by_band"]["lowpass"]["order_ok"]/finetuned["by_band"]["lowpass"]["total"] - baseline["by_band"]["lowpass"]["order_ok"]/baseline["by_band"]["lowpass"]["total"])*100:.1f}%'],
        ['HPF 阶数准确率',
         f'{baseline["by_band"]["highpass"]["order_ok"]/baseline["by_band"]["highpass"]["total"]*100:.1f}%',
         f'{finetuned["by_band"]["highpass"]["order_ok"]/finetuned["by_band"]["highpass"]["total"]*100:.1f}%',
         f'+{(finetuned["by_band"]["highpass"]["order_ok"]/finetuned["by_band"]["highpass"]["total"] - baseline["by_band"]["highpass"]["order_ok"]/baseline["by_band"]["highpass"]["total"])*100:.1f}%'],
        ['BPF 阶数准确率',
         f'{baseline["by_band"]["bandpass"]["order_ok"]/baseline["by_band"]["bandpass"]["total"]*100:.1f}%',
         f'{finetuned["by_band"]["bandpass"]["order_ok"]/finetuned["by_band"]["bandpass"]["total"]*100:.1f}%',
         f'+{(finetuned["by_band"]["bandpass"]["order_ok"]/finetuned["by_band"]["bandpass"]["total"] - baseline["by_band"]["bandpass"]["order_ok"]/baseline["by_band"]["bandpass"]["total"])*100:.1f}%'],
        ['参数键完整率',
         f'{baseline["keys_complete"]/b_total*100:.1f}%',
         f'{finetuned["keys_complete"]/f_total*100:.1f}%',
         f'+{(finetuned["keys_complete"]/f_total - baseline["keys_complete"]/b_total)*100:.1f}%'],
    ]

    # 200样本评估补充数据行
    row_data.append([
        '200样本评估-阶数准确率',
        '—',
        f'{summary["full_order_acc"]*100:.1f}%',
        '—'
    ])

    table = ax.table(cellText=row_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.3, 0.2, 0.25, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # 样式
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

    for i in range(1, len(row_data) + 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[i, j].set_facecolor('#ECEFF1')

        # 高亮提升幅度列
        improvement_text = row_data[i-1][3]
        if improvement_text != '—' and improvement_text != '+0.0%':
            try:
                val = float(improvement_text.replace('+', '').replace('%', ''))
                if val > 30:
                    table[i, 3].set_text_props(color='#D32F2F', fontweight='bold')
                elif val > 0:
                    table[i, 3].set_text_props(color='#388E3C', fontweight='bold')
            except:
                pass

    ax.set_title('模型性能评估综合结果', fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'results_table.png')
    fig.savefig(SAVE_DIR / 'results_table.pdf')
    plt.close(fig)
    print('[OK] 综合结果表格 -> results_table.png/pdf')


# ============ 图7: 数值误差箱线图 ============
def plot_numerical_errors(eval_results):
    """绘制数值参数预测误差箱线图"""
    details = eval_results['details']['full']

    # 收集所有数值误差
    error_keys = ['ripple_db', 'R0', 'La_target']
    freq_keys = ['f_center', 'bandwidth', 'fs_lower', 'fs_upper']

    all_errors = {}
    for key in error_keys + freq_keys:
        errs = []
        for d in details:
            if key in d.get('num_errs', {}):
                errs.append(d['num_errs'][key] * 100)  # 转为百分比
        if errs:
            all_errors[key] = errs

    key_labels = {
        'ripple_db': '纹波\n(ripple_db)',
        'R0': '阻抗\n(R0)',
        'La_target': '衰减\n(La_target)',
        'f_center': '中心频率\n(f_center)',
        'bandwidth': '带宽\n(bandwidth)',
        'fs_lower': '下阻带\n(fs_lower)',
        'fs_upper': '上阻带\n(fs_upper)'
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    data = []
    for key in error_keys + freq_keys:
        if key in all_errors:
            labels.append(key_labels.get(key, key))
            data.append(all_errors[key])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markerfacecolor='#EF5350', markersize=3, alpha=0.5))

    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#26C6DA', '#EC407A', '#8D6E63']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('相对误差 (%)', fontsize=12)
    ax.set_title('微调模型数值参数预测误差分布 (200样本)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 标注中位数
    medians = [np.median(d) for d in data]
    for i, (med, label) in enumerate(zip(medians, labels)):
        ax.text(i+1, med, f'{med:.4f}%', ha='center', va='bottom', fontsize=8, color='#333')

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'numerical_errors.png')
    fig.savefig(SAVE_DIR / 'numerical_errors.pdf')
    plt.close(fig)
    print('[OK] 数值误差箱线图 -> numerical_errors.png/pdf')


# ============ 图8: 训练配置信息卡片 ============
def plot_training_config():
    """绘制训练配置信息卡片"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    config_items = [
        ('基础模型', 'Qwen3-8B (8.19B params, 36 layers)'),
        ('微调方法', 'QLoRA (NF4 4-bit quantization)'),
        ('LoRA 配置', 'r=8, α=16, dropout=0.05, target=q_proj+v_proj'),
        ('可训练参数', '3.83M / 8.19B (0.047%)'),
        ('训练数据', '25,168 samples (中英双语混合)'),
        ('验证数据', '1,415 samples'),
        ('测试数据', '1,372 samples'),
        ('学习率', '1e-4 (cosine scheduler, warmup=5%)'),
        ('批大小', '1 × 16 grad_accum = 16 effective'),
        ('训练轮数', '2 epochs (3,146 steps)'),
        ('最终损失', '0.2785'),
        ('训练时长', '9小时24分钟 (RTX 4090 24GB)'),
        ('推理模板', 'qwen3_nothink (禁用思考模式)'),
        ('框架', 'LLaMA-Factory + PEFT 0.17.1'),
    ]

    y_start = 0.95
    y_step = 0.065

    ax.text(0.5, 1.0, '训练配置摘要', fontsize=16, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    for i, (key, val) in enumerate(config_items):
        y = y_start - i * y_step
        ax.text(0.05, y, f'● {key}:', fontsize=11, fontweight='bold',
                transform=ax.transAxes, verticalalignment='top', color='#1565C0')
        ax.text(0.35, y, val, fontsize=11,
                transform=ax.transAxes, verticalalignment='top', color='#333')

    # 边框
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.02",
                          facecolor='#FAFAFA', edgecolor='#90A4AE',
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    rect.set_zorder(0)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'training_config.png')
    fig.savefig(SAVE_DIR / 'training_config.pdf')
    plt.close(fig)
    print('[OK] 训练配置信息卡片 -> training_config.png/pdf')


# ============ MAIN ============
def main():
    print('=' * 60)
    print('论文图表生成脚本')
    print('=' * 60)
    print(f'输出目录: {SAVE_DIR}')
    print()

    # 加载数据
    print('[加载] 训练日志...')
    logs = load_trainer_log()
    print(f'  -> {len(logs)} 条记录')

    print('[加载] 基线对比数据...')
    comp_data = load_baseline_comparison()
    print(f'  -> baseline: {comp_data["baseline"]["total"]} 样本, finetuned: {comp_data["finetuned"]["total"]} 样本')

    print('[加载] 评估结果...')
    eval_results = load_eval_results()
    print(f'  -> {eval_results["summary"]["n_full"]} full + {eval_results["summary"]["n_followup_q"]} followup_q + {eval_results["summary"]["n_followup_r"]} followup_r')

    print('[加载] 训练数据...')
    train_data = load_train_data()
    print(f'  -> {len(train_data)} training samples')

    print()
    print('[生成图表]')

    # 生成所有图表
    plot_training_loss(logs)
    plot_baseline_comparison(comp_data)
    plot_order_accuracy_by_band(comp_data)
    plot_dataset_distribution(train_data)
    plot_evaluation_radar(comp_data, eval_results)
    plot_results_table(comp_data, eval_results)
    plot_numerical_errors(eval_results)
    plot_training_config()

    print()
    print('=' * 60)
    print(f'所有图表已生成到: {SAVE_DIR}')
    generated = list(SAVE_DIR.glob('*'))
    print(f'共 {len(generated)} 个文件:')
    for f in sorted(generated):
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name} ({size_kb:.1f} KB)')
    print('=' * 60)


if __name__ == '__main__':
    main()
