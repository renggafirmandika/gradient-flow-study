# Gradient Flow as the Primary Bottleneck in RNNs
### An Empirical Analysis Against Transformers on Long-Sequence Tasks

> COMP5329 / COMP4329 Deep Learning — USYD Deep Learning Course Conference 2026  
> Rengga Firmandika · Niramay Kachhadiya · Agni Karuhatty

---

## Overview

Transformers consistently outperform RNNs on long-sequence tasks, but the mechanistic reason is rarely isolated. This project empirically investigates whether **gradient flow degradation alone** accounts for the performance gap, using controlled ablations on a synthetic task where gradient failure is directly measurable.

We train four architectures — Vanilla RNN, LSTM, GRU, and Transformer — on the **selective copy task** and measure gradient health metrics alongside accuracy across six ablation studies.

### Key finding (preliminary)

At sequence length T=500 with the critical token at position k=1:
- All three RNN variants collapse to ~50% accuracy (random guessing)
- Input gradient norm at position k=1 is **exactly zero** for all RNN variants
- Transformer maintains 100% accuracy with non-zero input gradient at k=1

This confirms the central hypothesis: gradient signal vanishes completely across 499 timesteps in RNNs, making the task unsolvable. The Transformer's O(1) attention path eliminates this problem regardless of sequence length.

---

## Research Claims

The paper makes four testable claims, each tied to a specific study. Significance tests for all claims are run via `python significance_tests.py` after experiments are complete (requires `results/all_results.json`).

**Claim 1 — Gradient flow failure causes accuracy collapse (Studies 1 & 2)**  
RNN variants collapse to ~50% accuracy exactly when input gradient norm at position k reaches zero. Tested by comparing Transformer vs RNN/LSTM/GRU accuracy at k=1 via independent t-test. Expected: p < 0.001.

**Claim 2 — Gradient signal decays with distance from output (Study 2)**  
Input gradient norm at position k drops to near-zero as k moves further from the final timestep. Tested by comparing gradient norms at k=498 vs k=1 for each RNN variant. Expected: p < 0.01 for all three.

**Claim 3 — LSTM and GRU extend the gradient highway but do not eliminate degradation (Study 2)**  
LSTM/GRU maintain a non-zero gradient signal further back than Vanilla RNN, but eventually collapse at some critical k*. The threshold k* is identified empirically as the smallest k where mean accuracy drops below 60%. Pairwise t-tests compare LSTM vs RNN at k ∈ {100, 50, 25, 10} to find where they diverge.

**Claim 4 — Gradient path length, not architecture, is the causal mechanism (Study 5)**  
Adding a single attention head to a Vanilla RNN (AttentionRNN) restores an O(1) gradient path. If performance recovers to near-Transformer level, this causally isolates gradient path length as the bottleneck — not attention's representational power or parameter count. Tested by comparing AttentionRNN vs RNN and AttentionRNN vs Transformer at k=1.

---

## Project Structure

```
gradient-flow-study/
├── tasks/
│   └── selective_copy.py       # Synthetic task generator
├── models/
│   ├── rnn.py                  # Vanilla RNN classifier
│   ├── lstm.py                 # LSTM classifier
│   ├── gru.py                  # GRU classifier
│   ├── transformer.py          # Transformer encoder classifier
│   └── attention_rnn.py        # RNN augmented with single-head attention (Study 5)
├── utils/
│   └── gradient_metrics.py     # GradientTracker — parameter-level hooks
├── train.py                    # Training loop, eval, input gradient measurement
├── run_study1.py               # Study 1: Sequence length scaling
├── run_study2.py               # Study 2: Critical token position + LSTM threshold
├── run_study3.py               # Study 3: Gradient clipping
├── run_study4.py               # Study 4: Architecture depth
├── run_study5.py               # Study 5: Causal intervention (AttentionRNN)
├── run_study6.py               # Study 6: Permuted MNIST validation
├── run_ablation.py             # Runs all studies sequentially
├── plot_results.py             # Generates paper figures as PDF
├── significance_tests.py       # Statistical tests for all 4 claims
├── smoke_test.py               # Sanity check before running full sweep
└── results/
    ├── all_results.json        # Raw output — all studies merged into one file
    └── figures/                # PDF figures for the paper
```

---

## Setup

```bash
git clone <repo-url>
cd gradient-flow-study
pip install torch torchvision numpy matplotlib scipy
```

No other dependencies. Does not require a GPU (but strongly recommended for the full sweep — CPU at T=500 is slow).

---

## Quickstart

### 1. Sanity check

```bash
python smoke_test.py
```

Expected: all 4 models achieve >70% accuracy at T=50, k=48. All gradient metrics are non-zero.

### 2. Run individual studies

Each study can be run independently. Results are merged into `results/all_results.json`:

```bash
python run_study1.py   # ~30 min GPU
python run_study2.py   # ~1–2 hr GPU  (longest — most k values × 3 seeds)
python run_study3.py   # ~10 min GPU
python run_study4.py   # ~20 min GPU
python run_study5.py   # ~1 hr GPU
python run_study6.py   # ~30 min GPU  (downloads MNIST on first run)
```

### 3. Generate figures

```bash
python plot_results.py
```

Produces PDF figures in `results/figures/` and prints a LaTeX summary table to stdout.

### 4. Run significance tests

```bash
python significance_tests.py
```

Prints p-values for all 4 research claims. Run after all studies are complete.

---

## The Task

**Selective Copy Task** — a standard synthetic benchmark for long-range dependency.

- Input: sequence of T integer tokens
- Token at position k is the label (0 or 1)
- All other tokens are noise drawn from {2,...,9} — uniquely distinguishable from the label token
- Output: predict the value of the token at position k

**Why this task?** If a model cannot propagate gradient signal back to position k, it literally cannot learn the task. Accuracy degrades to 50% (chance) in direct proportion to gradient failure. This makes gradient health and task performance directly comparable.

| Parameter | Value |
|---|---|
| Vocabulary size | 10 |
| Label classes | 2 (binary) |
| Chance accuracy | 50% |
| Training samples | 8,000 |
| Validation samples | 2,000 |

---

## Architectures

All models use `embed_dim=64`. Parameter counts are reported to acknowledge the known capacity discrepancy.

| Architecture | Hidden dim | Approx params | Key property |
|---|---|---|---|
| Vanilla RNN | 128 | ~25K | Baseline — worst gradient flow |
| LSTM | 128 | ~100K | Cell state creates partial gradient highway |
| GRU | 128 | ~75K | Lighter than LSTM, similar protection |
| Transformer | 64 embed, 4 heads, 1 layer | ~200K+ | O(1) gradient path via self-attention |
| AttentionRNN | 128 + single attention head | ~50K | Study 5 only — surgical gradient path restoration |

> **Note on parameter mismatch:** The Transformer has significantly more parameters than the RNN variants at matched hidden dimensions. This is a known confound acknowledged in the paper. The key counter-argument: if input gradient norm at position k is exactly zero, no amount of additional capacity can learn from that position — capacity is irrelevant when the learning signal is absent.

---

## Ablation Studies

Each study is run as a standalone script and saves its output into `results/all_results.json`.

### Study 1 — Sequence Length Scaling
`run_study1.py`  
Vary T ∈ {50, 100, 200, 500, 1000} with critical token near the end (k = T−2). Tests how gradient health and accuracy degrade as sequences grow longer. All 4 architectures.

### Study 2 — Critical Token Position + LSTM Threshold
`run_study2.py`  
Fix T=500. Vary k ∈ {498, 400, 350, 300, 250, 200, 150, 100, 50, 25, 10, 1}. Directly measures input gradient norm at position k for all architectures. The finer k resolution in the 50–300 range is designed to identify the empirical threshold k* where LSTM and GRU gradient highways collapse — quantifying how much further each architecture's gradient can propagate compared to Vanilla RNN.

### Study 3 — Gradient Clipping
`run_study3.py`  
Fix architecture=RNN, T=500, k=250. Vary clip threshold ∈ {None, 10, 5, 1, 0.1}. Isolates the exploding gradient failure mode and shows that fixing explosion does not rescue vanishing — confirming the bottleneck is vanishing, not exploding.

### Study 4 — Architecture Depth
`run_study4.py`  
Vary L ∈ {1, 2, 4} stacked layers at fixed T=200, k=100. Tests whether adding depth compounds gradient degradation in RNNs (expected: yes) vs Transformer (expected: no effect on temporal gradient path).

### Study 5 — Causal Intervention: Gradient Path Restoration
`run_study5.py`  
Fix T=500. Vary k ∈ {498, 400, 250, 100, 10, 1}. Compares Vanilla RNN, AttentionRNN (RNN + single-head attention), and Transformer. If adding only the O(1) attention path to an RNN recovers near-Transformer accuracy and restores non-zero gradient at k=1, this causally isolates gradient path length as the bottleneck — ruling out that Transformer's advantage comes from its representational power or parameter count.

### Study 6 — Real-World Validation: Permuted MNIST
`run_study6.py`  
Validates that the gradient failure patterns observed on the synthetic task generalise to real sequential data. MNIST images (28×28) are flattened and fed one pixel at a time, with a fixed random permutation applied to break local spatial correlations. This creates a genuine long-range dependency problem at T=784. Input gradient norm is measured at positions {1, 100, 300, 500, 783} to show the gradient decay curve across the full sequence on real data. All 4 architectures, 10-class classification (chance = 10%).

---

## Gradient Metrics

Two types of gradients are tracked, serving different purposes:

**Parameter gradients** (via `GradientTracker` in `utils/gradient_metrics.py`)  
Captured using `register_hook()` on all model parameters after each `backward()` call.

| Metric | Description |
|---|---|
| `mean_grad_norm` | Average gradient magnitude across all parameters |
| `vanishing_ratio` | Fraction of parameters with gradient norm < 1e-4 |
| `max_grad_norm` | Maximum gradient (exploding signal indicator) |

**Input gradient at position k** (via `get_input_gradient_at_position` in `train.py`)  
Measures `∂loss/∂embedding[k]` — how much the loss responded to the token at position k. Requires `retain_grad()` on the embedding output since PyTorch discards intermediate tensor gradients by default.

```python
embedded = model.embedding(sequence)
embedded.retain_grad()   # must call before backward()
# ... forward pass ...
loss.backward()
grad_norm_at_k = embedded.grad[0, critical_pos, :].norm().item()
```

This metric is zero when the model has completely ignored position k — the direct empirical signature of vanishing gradients.

---

## Reproducibility

### Multiple seeds

All studies are run with 3 fixed seeds: `[42, 123, 456]`. Each experiment configuration is repeated 3 times with different random initialisations. Results are aggregated before plotting — all figures show **mean ± 1 std** across seeds as error bars.

The permutation in Study 6 uses a separate fixed seed (0) and does not vary across training seeds — the pixel ordering is identical across all runs so results are comparable.

### Significance tests

`significance_tests.py` loads `results/all_results.json` and runs independent-samples t-tests across the 3 seed runs for each comparison. Output format:

```
Transformer vs RNN at k=1: p=0.0003 ***  (mean A=0.998, mean B=0.501)
```

Significance thresholds: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant.

The script tests all 4 research claims automatically and also identifies the empirical LSTM failure threshold k* from Study 2 data.

---

## Compute

| Configuration | Estimated runtime |
|---|---|
| Smoke test (T=50, 3 epochs, CPU) | ~30 seconds |
| Study 1 (5 T values × 4 archs × 3 seeds, GPU) | ~30 min |
| Study 2 (12 k values × 4 archs × 3 seeds, GPU) | ~1–2 hr |
| Study 3 (5 clip values × 3 seeds, GPU) | ~10 min |
| Study 4 (3 depths × 4 archs × 3 seeds, GPU) | ~20 min |
| Study 5 (6 k values × 3 archs × 3 seeds, GPU) | ~1 hr |
| Study 6 (4 archs × 3 seeds, GPU) | ~30 min |
| **Full sweep, all studies (GPU)** | **~4–6 hours** |
| Full sweep (CPU) | Not recommended at T=500+ |

Kaggle free tier (30h/week GPU) is sufficient. Studies can be run independently — results accumulate in `all_results.json`.

---

## Results Layout

Each `run_study*.py` merges its output into `results/all_results.json`:

```
results/all_results.json
{
  "study_1": [ { "arch": "rnn", "seq_len": 50, "seed": 42, "final_val_acc": ..., "history": [...] }, ... ],
  "study_2": [ { "arch": "rnn", "critical_pos": 1, "seed": 42, "input_grad_norm": 0.0, ... }, ... ],
  "study_3": [ { "arch": "rnn", "clip": 5.0, "seed": 42, "final_val_acc": ..., ... }, ... ],
  "study_4": [ { "arch": "rnn", "depth": 2, "seed": 42, "final_val_acc": ..., ... }, ... ],
  "study_5": [ { "arch": "attention_rnn", "critical_pos": 1, "seed": 42, ... }, ... ],
  "study_6": [ { "arch": "rnn", "seq_len": 784, "seed": 42, "grad_at_positions": {...}, ... }, ... ]
}
```

`plot_results.py` reads this file and produces:

| File | Content |
|---|---|
| `figures/study1_sequence_length.pdf` | Accuracy + gradient norm vs T (mean ± std) |
| `figures/study2_critical_position.pdf` | Input gradient norm + accuracy vs k (mean ± std) |
| `figures/study3_clipping.pdf` | Accuracy + gradient norm vs clip threshold (mean ± std) |
| `figures/study4_depth.pdf` | Accuracy vs number of layers (mean ± std) |
| `figures/study5_intervention.pdf` | Gradient norm + accuracy: RNN vs AttentionRNN vs Transformer |

---

## Paper

Submitted to the USYD Deep Learning Course Conference 2026 via OpenReview.  
6–8 pages, double-column, DLCC LaTeX template.  
**Final submission deadline: 24 May 2026, 23:59 Sydney time.**

---

## Citation

```
@article{firmandika2026gradient,
  title={Gradient Flow as the Primary Bottleneck in Recurrent Neural Networks:
         An Empirical Analysis Against Transformers on Long-Sequence Tasks},
  author={Firmandika, Rengga and Kachhadiya, Niramay and Karuhatty, Agni},
  journal={USYD Deep Learning Course Conference},
  year={2026}
}
```
