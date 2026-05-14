# significance_tests.py
# Run after all studies are complete. Prints p-values for the paper's key claims.
# Requires: results/all_results.json (produced by run_study*.py)

import json
import numpy as np
from scipy.stats import ttest_ind

with open('results/all_results.json') as f:
    data = json.load(f)

def get_accs(study, filters):
    """Extract final_val_acc from a study for all entries matching filters dict."""
    return [
        r['final_val_acc'] for r in data[study]
        if all(r.get(k) == v for k, v in filters.items())
    ]

def get_grads(study, filters):
    """Extract input_grad_norm from a study for all entries matching filters dict."""
    return [
        r['input_grad_norm'] for r in data[study]
        if all(r.get(k) == v for k, v in filters.items())
    ]

def ttest(a, b, label):
    if len(a) < 2 or len(b) < 2:
        print(f"  {label}: insufficient samples (need >=2 per group)")
        return
    stat, p = ttest_ind(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label}: p={p:.4f} {sig}  "
          f"(mean A={np.mean(a):.3f}, mean B={np.mean(b):.3f})")

# ── Claim 1: Transformer outperforms RNN variants at k=1 (Study 2) ───────────

print("\n=== Claim 1: Transformer vs RNN variants at k=1, T=500 (Study 2) ===")
trans_accs = get_accs('study_2', {'arch': 'transformer', 'critical_pos': 1})
for arch in ['rnn', 'lstm', 'gru']:
    arch_accs = get_accs('study_2', {'arch': arch, 'critical_pos': 1})
    ttest(trans_accs, arch_accs, f"Transformer vs {arch.upper()}")

# ── Claim 2: Gradient signal collapses at k=1 vs k=498 (Study 2) ─────────────

print("\n=== Claim 2: Gradient collapse at k=1 vs k=498 for RNN variants (Study 2) ===")
for arch in ['rnn', 'lstm', 'gru']:
    grads_near  = get_grads('study_2', {'arch': arch, 'critical_pos': 498})
    grads_far   = get_grads('study_2', {'arch': arch, 'critical_pos': 1})
    ttest(grads_near, grads_far, f"{arch.upper()} grad@k=498 vs grad@k=1")

# ── Claim 3: LSTM outlasts RNN — threshold analysis (Study 2) ────────────────
# Find the k where LSTM mean accuracy first drops below 0.6 and compare to RNN

print("\n=== Claim 3: LSTM failure threshold vs RNN (Study 2) ===")
from collections import defaultdict

def mean_acc_by_k(study, arch):
    by_k = defaultdict(list)
    for r in data[study]:
        if r['arch'] == arch:
            by_k[r['critical_pos']].append(r['final_val_acc'])
    return {k: np.mean(v) for k, v in sorted(by_k.items())}

rnn_by_k  = mean_acc_by_k('study_2', 'rnn')
lstm_by_k = mean_acc_by_k('study_2', 'lstm')
gru_by_k  = mean_acc_by_k('study_2', 'gru')

THRESHOLD = 0.6
for arch, by_k in [('RNN', rnn_by_k), ('LSTM', lstm_by_k), ('GRU', gru_by_k)]:
    # Find smallest k where accuracy >= threshold (i.e. gradient still flows)
    surviving = [k for k, acc in sorted(by_k.items()) if acc >= THRESHOLD]
    if surviving:
        print(f"  {arch}: gradient highway survives to k={min(surviving)} "
              f"(last k with acc >= {THRESHOLD})")
    else:
        print(f"  {arch}: accuracy never reaches {THRESHOLD} at any k")

# Compare LSTM vs RNN at mid-range k values where LSTM still works but RNN fails
print("\n  Pairwise t-tests at selected k values:")
for k in [100, 50, 25, 10]:
    rnn_accs  = get_accs('study_2', {'arch': 'rnn',  'critical_pos': k})
    lstm_accs = get_accs('study_2', {'arch': 'lstm', 'critical_pos': k})
    if rnn_accs and lstm_accs:
        ttest(lstm_accs, rnn_accs, f"LSTM vs RNN at k={k}")

# ── Claim 4: Gradient path restoration (Study 5) ─────────────────────────────

print("\n=== Claim 4: Attention augmentation restores performance at k=1 (Study 5) ===")
rnn_accs  = get_accs('study_5', {'arch': 'rnn',          'critical_pos': 1})
attn_accs = get_accs('study_5', {'arch': 'attention_rnn', 'critical_pos': 1})
trans_accs = get_accs('study_5', {'arch': 'transformer',  'critical_pos': 1})
ttest(attn_accs, rnn_accs,   "AttentionRNN vs RNN at k=1")
ttest(attn_accs, trans_accs, "AttentionRNN vs Transformer at k=1")



print("\nDone. Use *, **, *** thresholds: p<0.05, p<0.01, p<0.001. 'ns' = not significant.")
