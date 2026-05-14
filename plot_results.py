# plot_results.py
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────

ARCH_STYLE = {
    'rnn':          {'color': '#E24B4A', 'marker': 'o', 'label': 'Vanilla RNN'},
    'lstm':         {'color': '#378ADD', 'marker': 's', 'label': 'LSTM'},
    'gru':          {'color': '#1D9E75', 'marker': '^', 'label': 'GRU'},
    'transformer':  {'color': '#7F77DD', 'marker': 'D', 'label': 'Transformer'},
    'attention_rnn':{'color': '#F5A623', 'marker': 'P', 'label': 'RNN + Attention'},
}

def style(arch):
    return ARCH_STYLE[arch]

# ── Aggregation helper ────────────────────────────────────────────────────────

def aggregate(results, group_keys, value_keys):
    """
    Group results by group_keys, compute mean and std for each value_key.
    Returns: {group_tuple: {value_key: (mean, std), ...}}
    """
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = tuple(r[k] for k in group_keys)
        for vk in value_keys:
            val = r.get(vk)
            if val is not None:
                groups[key][vk].append(val)
    return {
        k: {vk: (np.mean(vs), np.std(vs)) for vk, vs in vals.items()}
        for k, vals in groups.items()
    }

# ── Load results ──────────────────────────────────────────────────────────────

with open('results/all_results.json') as f:
    data = json.load(f)

Path('results/figures').mkdir(exist_ok=True)

# ── Figure 1: Study 1 — sequence length ──────────────────────────────────────

def plot_study1():
    agg = aggregate(data['study_1'], ['arch', 'seq_len'],
                    ['final_val_acc', 'input_grad_norm'])

    by_arch = {a: {'T': [], 'acc_mean': [], 'acc_std': [], 'grad_mean': [], 'grad_std': []}
               for a in ARCH_STYLE}

    for (arch, T), vals in sorted(agg.items(), key=lambda x: x[0][1]):
        if arch not in by_arch:
            continue
        by_arch[arch]['T'].append(T)
        acc_m, acc_s   = vals.get('final_val_acc', (0, 0))
        grad_m, grad_s = vals.get('input_grad_norm', (0, 0))
        by_arch[arch]['acc_mean'].append(acc_m)
        by_arch[arch]['acc_std'].append(acc_s)
        by_arch[arch]['grad_mean'].append(grad_m)
        by_arch[arch]['grad_std'].append(grad_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch, vals in by_arch.items():
        if not vals['T']:
            continue
        s = style(arch)
        T        = np.array(vals['T'])
        acc_m    = np.array(vals['acc_mean'])
        acc_s    = np.array(vals['acc_std'])
        grad_m   = np.array(vals['grad_mean'])
        grad_s   = np.array(vals['grad_std'])

        ax1.errorbar(T, acc_m, yerr=acc_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)
        ax2.errorbar(T, grad_m, yerr=grad_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)

    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance (50%)')
    ax1.set_xlabel('Sequence length T')
    ax1.set_ylabel('Validation accuracy')
    ax1.set_title('Accuracy vs sequence length')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.4, 1.05)

    ax2.set_xlabel('Sequence length T')
    ax2.set_ylabel('Mean gradient norm')
    ax2.set_title('Gradient norm vs sequence length')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('results/figures/study1_sequence_length.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study1_sequence_length.pdf")

# ── Figure 2: Study 2 — critical position ────────────────────────────────────

def plot_study2():
    agg = aggregate(data['study_2'], ['arch', 'critical_pos'],
                    ['final_val_acc', 'input_grad_norm'])

    by_arch = {a: {'k': [], 'acc_mean': [], 'acc_std': [], 'grad_mean': [], 'grad_std': []}
               for a in ARCH_STYLE}

    for (arch, k), vals in sorted(agg.items(), key=lambda x: x[0][1]):
        if arch not in by_arch:
            continue
        by_arch[arch]['k'].append(k)
        acc_m, acc_s   = vals.get('final_val_acc', (0, 0))
        grad_m, grad_s = vals.get('input_grad_norm', (0, 0))
        by_arch[arch]['acc_mean'].append(acc_m)
        by_arch[arch]['acc_std'].append(acc_s)
        by_arch[arch]['grad_mean'].append(grad_m)
        by_arch[arch]['grad_std'].append(grad_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch, vals in by_arch.items():
        if not vals['k']:
            continue
        s = style(arch)
        k      = np.array(vals['k'])
        acc_m  = np.array(vals['acc_mean'])
        acc_s  = np.array(vals['acc_std'])
        grad_m = np.array(vals['grad_mean'])
        grad_s = np.array(vals['grad_std'])

        # Replace zeros for log scale
        grad_m_plot = np.where(grad_m == 0, 1e-10, grad_m)

        ax1.errorbar(k, grad_m_plot, yerr=grad_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)
        ax2.errorbar(k, acc_m, yerr=acc_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)

    ax1.set_xlabel('Critical token position k')
    ax1.set_ylabel('Input gradient norm at position k')
    ax1.set_title('Gradient signal vs token position')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)

    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax2.set_xlabel('Critical token position k')
    ax2.set_ylabel('Validation accuracy')
    ax2.set_title('Accuracy vs critical token position')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig('results/figures/study2_critical_position.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study2_critical_position.pdf")

# ── Figure 3: Study 3 — gradient clipping ────────────────────────────────────

def plot_study3():
    agg = aggregate(data['study_3'], ['clip'],
                    ['final_val_acc', 'input_grad_norm'])

    clips_sorted = sorted(agg.keys(), key=lambda x: x[0] if x[0] is not None else 999)
    clip_labels  = [str(c[0]) if c[0] is not None else 'None' for c in clips_sorted]
    acc_means    = [agg[c]['final_val_acc'][0] for c in clips_sorted]
    acc_stds     = [agg[c]['final_val_acc'][1] for c in clips_sorted]
    grad_means   = [agg[c].get('input_grad_norm', (0, 0))[0] for c in clips_sorted]
    grad_stds    = [agg[c].get('input_grad_norm', (0, 0))[1] for c in clips_sorted]

    x = np.arange(len(clip_labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    s = style('rnn')
    ax1.bar(x, acc_means, yerr=acc_stds, color=s['color'], alpha=0.8, capsize=4)
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax1.set_xticks(x); ax1.set_xticklabels(clip_labels)
    ax1.set_xlabel('Gradient clip threshold')
    ax1.set_ylabel('Validation accuracy')
    ax1.set_title('Vanilla RNN: clipping vs accuracy')
    ax1.set_ylim(0.4, 1.05)

    ax2.bar(x, grad_means, yerr=grad_stds, color=s['color'], alpha=0.8, capsize=4)
    ax2.set_xticks(x); ax2.set_xticklabels(clip_labels)
    ax2.set_xlabel('Gradient clip threshold')
    ax2.set_ylabel('Input gradient norm')
    ax2.set_title('Vanilla RNN: clipping vs gradient norm')
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.savefig('results/figures/study3_clipping.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study3_clipping.pdf")

# ── Figure 4: Study 4 — depth ─────────────────────────────────────────────────

def plot_study4():
    agg = aggregate(data['study_4'], ['arch', 'depth'], ['final_val_acc'])

    by_arch = {a: {'depth': [], 'acc_mean': [], 'acc_std': []} for a in ARCH_STYLE}

    for (arch, depth), vals in sorted(agg.items(), key=lambda x: x[0][1]):
        if arch not in by_arch:
            continue
        by_arch[arch]['depth'].append(depth)
        acc_m, acc_s = vals.get('final_val_acc', (0, 0))
        by_arch[arch]['acc_mean'].append(acc_m)
        by_arch[arch]['acc_std'].append(acc_s)

    fig, ax = plt.subplots(figsize=(6, 4))

    for arch, vals in by_arch.items():
        if not vals['depth']:
            continue
        s = style(arch)
        ax.errorbar(vals['depth'], vals['acc_mean'], yerr=vals['acc_std'],
                    color=s['color'], marker=s['marker'],
                    label=s['label'], linewidth=1.5, capsize=3)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('Accuracy vs architecture depth')
    ax.set_xticks([1, 2, 4])
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig('results/figures/study4_depth.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study4_depth.pdf")

# ── Figure 5: Study 5 — gradient path restoration ────────────────────────────

def plot_study5():
    if 'study_5' not in data:
        print("Study 5 results not found — run run_study5.py first.")
        return

    archs = ['rnn', 'attention_rnn', 'transformer']
    agg = aggregate(data['study_5'], ['arch', 'critical_pos'],
                    ['final_val_acc', 'input_grad_norm'])

    by_arch = {a: {'k': [], 'acc_mean': [], 'acc_std': [], 'grad_mean': [], 'grad_std': []}
               for a in archs}

    for (arch, k), vals in sorted(agg.items(), key=lambda x: x[0][1]):
        if arch not in by_arch:
            continue
        by_arch[arch]['k'].append(k)
        acc_m, acc_s   = vals.get('final_val_acc', (0, 0))
        grad_m, grad_s = vals.get('input_grad_norm', (0, 0))
        by_arch[arch]['acc_mean'].append(acc_m)
        by_arch[arch]['acc_std'].append(acc_s)
        by_arch[arch]['grad_mean'].append(grad_m)
        by_arch[arch]['grad_std'].append(grad_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch in archs:
        vals = by_arch[arch]
        if not vals['k']:
            continue
        s = style(arch)
        k      = np.array(vals['k'])
        acc_m  = np.array(vals['acc_mean'])
        acc_s  = np.array(vals['acc_std'])
        grad_m = np.array(vals['grad_mean'])
        grad_s = np.array(vals['grad_std'])

        grad_m_plot = np.where(grad_m == 0, 1e-10, grad_m)

        ax1.errorbar(k, grad_m_plot, yerr=grad_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)
        ax2.errorbar(k, acc_m, yerr=acc_s, color=s['color'], marker=s['marker'],
                     label=s['label'], linewidth=1.5, capsize=3)

    ax1.set_xlabel('Critical token position k')
    ax1.set_ylabel('Input gradient norm at position k')
    ax1.set_title('Gradient signal: RNN vs RNN+Attention vs Transformer')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)

    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax2.set_xlabel('Critical token position k')
    ax2.set_ylabel('Validation accuracy')
    ax2.set_title('Accuracy restored by attention augmentation')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig('results/figures/study5_intervention.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study5_intervention.pdf")

# ── Table 1: Summary ──────────────────────────────────────────────────────────

def print_summary_table():
    agg = aggregate(
        [r for r in data['study_1'] if r['seq_len'] == 500],
        ['arch'],
        ['final_val_acc', 'input_grad_norm']
    )
    print("\n% Table 1 — paste into LaTeX")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Architecture & Val Accuracy (mean $\pm$ std) & Input Grad Norm (mean $\pm$ std) \\")
    print(r"\midrule")
    for (arch,), vals in sorted(agg.items(), key=lambda x: -x[1]['final_val_acc'][0]):
        acc_m, acc_s   = vals['final_val_acc']
        grad_m, grad_s = vals.get('input_grad_norm', (0, 0))
        print(f"{arch.upper()} & "
              f"${acc_m:.3f} \\pm {acc_s:.3f}$ & "
              f"${grad_m:.2e} \\pm {grad_s:.2e}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    plot_study1()
    plot_study2()
    plot_study3()
    plot_study4()
    if 'study_5' in data:
        plot_study5()
    print_summary_table()
