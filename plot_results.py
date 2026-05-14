# plot_results.py
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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

# ── Load results ──────────────────────────────────────────────────────────────

with open('results/all_results.json') as f:
    data = json.load(f)

Path('results/figures').mkdir(exist_ok=True)

# ── Figure 1: Study 1 — sequence length ──────────────────────────────────────

def plot_study1():
    results = data['study_1']
    # Group by arch
    by_arch = {a: {'T': [], 'acc': [], 'grad': []} for a in ARCH_STYLE}
    for r in results:
        a = r['arch']
        by_arch[a]['T'].append(r['seq_len'])
        by_arch[a]['acc'].append(r['final_val_acc'])
        by_arch[a]['grad'].append(r['history'][-1]['mean_grad_norm'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch, vals in by_arch.items():
        s = style(arch)
        idx = np.argsort(vals['T'])
        T   = np.array(vals['T'])[idx]
        acc = np.array(vals['acc'])[idx]
        grad = np.array(vals['grad'])[idx]

        ax1.plot(T, acc,  color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)
        ax2.plot(T, grad, color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)

    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance (50%)')
    ax1.set_xlabel('Sequence length T')
    ax1.set_ylabel('Validation accuracy')
    ax1.set_title('Accuracy vs sequence length')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.4, 1.05)

    ax2.set_xlabel('Sequence length T')
    ax2.set_ylabel('Mean gradient norm')
    ax2.set_title('Gradient norm vs sequence length')
    ax2.set_yscale('log')          # log scale — norms span several orders of magnitude
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('results/figures/study1_sequence_length.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study1_sequence_length.pdf")

# ── Figure 2: Study 2 — critical position ────────────────────────────────────

def plot_study2():
    results = data['study_2']
    by_arch = {a: {'k': [], 'input_grad': [], 'acc': []} for a in ARCH_STYLE}
    for r in results:
        a = r['arch']
        by_arch[a]['k'].append(r['critical_pos'])
        by_arch[a]['input_grad'].append(r['input_grad_norm'])
        by_arch[a]['acc'].append(r['final_val_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch, vals in by_arch.items():
        s = style(arch)
        idx = np.argsort(vals['k'])
        k    = np.array(vals['k'])[idx]
        ig   = np.array(vals['input_grad'])[idx]
        acc  = np.array(vals['acc'])[idx]

        ax1.plot(k, ig,  color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)
        ax2.plot(k, acc, color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)

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
    results = data['study_3']
    clips = []
    accs  = []
    grads = []

    for r in sorted(results, key=lambda x: x['clip'] or 999):
        clips.append(str(r['clip']) if r['clip'] else 'None')
        accs.append(r['final_val_acc'])
        grads.append(r['history'][-1]['mean_grad_norm'])

    x = np.arange(len(clips))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    s = style('rnn')
    ax1.bar(x, accs,  color=s['color'], alpha=0.8)
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax1.set_xticks(x); ax1.set_xticklabels(clips)
    ax1.set_xlabel('Gradient clip threshold')
    ax1.set_ylabel('Validation accuracy')
    ax1.set_title('Vanilla RNN: clipping vs accuracy')
    ax1.set_ylim(0.4, 1.05)

    ax2.bar(x, grads, color=s['color'], alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(clips)
    ax2.set_xlabel('Gradient clip threshold')
    ax2.set_ylabel('Mean gradient norm')
    ax2.set_title('Vanilla RNN: clipping vs gradient norm')
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.savefig('results/figures/study3_clipping.pdf', bbox_inches='tight')
    plt.close()
    print("Saved study3_clipping.pdf")

# ── Figure 4: Study 4 — depth ─────────────────────────────────────────────────

def plot_study4():
    results = data['study_4']
    by_arch = {a: {'depth': [], 'acc': []} for a in ARCH_STYLE}
    for r in results:
        a = r['arch']
        by_arch[a]['depth'].append(r['depth'])
        by_arch[a]['acc'].append(r['final_val_acc'])

    fig, ax = plt.subplots(figsize=(6, 4))

    for arch, vals in by_arch.items():
        s = style(arch)
        idx = np.argsort(vals['depth'])
        d   = np.array(vals['depth'])[idx]
        acc = np.array(vals['acc'])[idx]
        ax.plot(d, acc, color=s['color'], marker=s['marker'],
                label=s['label'], linewidth=1.5)

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

# ── Table 1: Summary ──────────────────────────────────────────────────────────

def print_summary_table():
    """
    Prints a LaTeX-ready table of final val accuracy and vanishing ratio
    at T=500 from Study 1.
    """
    results = [r for r in data['study_1'] if r['seq_len'] == 500]
    print("\n% Table 1 — paste into LaTeX")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Architecture & Val Accuracy & Vanishing Ratio & Mean Grad Norm \\")
    print(r"\midrule")
    for r in sorted(results, key=lambda x: x['final_val_acc'], reverse=True):
        last = r['history'][-1]
        print(f"{r['arch'].upper()} & "
              f"{r['final_val_acc']:.3f} & "
              f"{last.get('vanishing_ratio', 0):.3f} & "
              f"{last.get('mean_grad_norm', 0):.2e} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

# ── Figure 5: Study 5 — gradient path restoration ────────────────────────────

def plot_study5():
    if 'study_5' not in data:
        print("Study 5 results not found — run run_study5.py first.")
        return

    results = data['study_5']
    archs = ['rnn', 'attention_rnn', 'transformer']
    by_arch = {a: {'k': [], 'input_grad': [], 'acc': []} for a in archs}

    for r in results:
        a = r['arch']
        if a in by_arch:
            by_arch[a]['k'].append(r['critical_pos'])
            by_arch[a]['input_grad'].append(r['input_grad_norm'])
            by_arch[a]['acc'].append(r['final_val_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for arch in archs:
        vals = by_arch[arch]
        s    = style(arch)
        idx  = np.argsort(vals['k'])
        k    = np.array(vals['k'])[idx]
        ig   = np.array(vals['input_grad'])[idx]
        acc  = np.array(vals['acc'])[idx]

        # replace zeros with a small value for log scale
        ig = np.where(ig == 0, 1e-10, ig)

        ax1.plot(k, ig,  color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)
        ax2.plot(k, acc, color=s['color'], marker=s['marker'],
                 label=s['label'], linewidth=1.5)

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

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    plot_study1()
    plot_study2()
    plot_study3()
    plot_study4()
    if 'study_5' in data:
        plot_study5()
    print_summary_table()