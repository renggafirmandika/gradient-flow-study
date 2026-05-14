# Gradient Flow as the Primary Bottleneck in RNNs
### An Empirical Analysis Against Transformers on Long-Sequence Tasks

> COMP5329 / COMP4329 Deep Learning — USYD Deep Learning Course Conference 2026  
> Rengga Firmandika · Niramay Kachhadiya · Agni Karuhatty

---

## Overview

Transformers consistently outperform RNNs on long-sequence tasks, but the mechanistic reason is rarely isolated. This project empirically investigates whether **gradient flow degradation alone** accounts for the performance gap, using controlled ablations on a synthetic task where gradient failure is directly measurable.

We train four architectures — Vanilla RNN, LSTM, GRU, and Transformer — on the **selective copy task** and measure gradient health metrics alongside accuracy across four ablation studies.

### Key finding (preliminary)

At sequence length T=500 with the critical token at position k=1:
- All three RNN variants collapse to ~50% accuracy (random guessing)
- Input gradient norm at position k=1 is **exactly zero** for all RNN variants
- Transformer maintains 100% accuracy with non-zero input gradient at k=1

This confirms the central hypothesis: gradient signal vanishes completely across 499 timesteps in RNNs, making the task unsolvable. The Transformer's O(1) attention path eliminates this problem regardless of sequence length.

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
│   └── transformer.py          # Transformer encoder classifier
├── utils/
│   └── gradient_metrics.py     # GradientTracker — parameter-level hooks
├── train.py                    # Training loop, eval, input gradient measurement
├── run_ablation.py             # Runs all 4 studies, saves results/all_results.json
├── plot_results.py             # Generates paper figures as PDF
├── smoke_test.py               # Sanity check before running full sweep
└── results/
    ├── all_results.json        # Raw output from run_ablation.py
    └── figures/                # PDF figures for the paper
```

---

## Setup

```bash
git clone <repo-url>
cd gradient-flow-study
pip install torch numpy matplotlib
```

No other dependencies. Does not require a GPU (but strongly recommended for the full sweep — CPU at T=500 is slow).

---

## Quickstart

### 1. Sanity check

Before running the full sweep, verify everything works end to end:

```bash
python smoke_test.py
```

Expected output: all 4 models achieve >70% accuracy at T=50, k=48. All gradient metrics are non-zero.

### 2. Stress test

Verify gradient degradation actually appears:

In `smoke_test.py`, change:
```python
SEQ_LEN  = 500
CRIT_POS = 1
```

Expected output: RNN/LSTM/GRU collapse to ~50% accuracy with near-zero input gradient at k. Transformer stays at 100%.

### 3. Full ablation sweep

```bash
python run_ablation.py
```

Runs all 4 studies and saves output to `results/all_results.json`. Runtime varies significantly by hardware — see note on compute below.

### 4. Generate figures

```bash
python plot_results.py
```

Produces 4 PDF figures in `results/figures/` and prints a LaTeX-ready summary table to stdout.

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

> **Note on parameter mismatch:** The Transformer has significantly more parameters than the RNN variants at matched hidden dimensions. This is a known confound acknowledged in the paper. Results are reported with parameter counts, and the Discussion addresses this limitation explicitly.

---

## Ablation Studies

### Study 1 — Sequence Length Scaling
Vary T ∈ {50, 100, 200, 500} with critical token near the end (k = T−2). Tests how gradient health and accuracy degrade as sequences grow longer.

### Study 2 — Critical Token Position
Fix T=500. Vary k ∈ {498, 400, 250, 100, 10, 1}. Directly measures input gradient norm at position k — the most mechanistically precise metric in the paper.

### Study 3 — Gradient Clipping
Fix architecture=RNN, T=500. Vary clip threshold ∈ {None, 10, 5, 1, 0.1}. Isolates exploding gradient failure mode and shows the tradeoff between explosion and over-clipping.

### Study 4 — Architecture Depth
Vary L ∈ {1, 2, 4} stacked layers at fixed T=200. Tests whether depth compounds gradient degradation in RNNs.

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

## Compute

| Configuration | Estimated runtime |
|---|---|
| Smoke test (T=50, 3 epochs, CPU) | ~30 seconds |
| Stress test (T=500, 3 epochs, CPU) | ~5 minutes |
| Full sweep, all studies (GPU) | ~2–4 hours |
| Full sweep, all studies (CPU) | Not recommended at T=500+ |

Kaggle free tier (30h/week GPU) is sufficient for the full sweep. Recommended: cap T at 500 if on CPU.

---

## Results Layout

`run_ablation.py` saves one JSON file:

```
results/all_results.json
{
  "study_1": [ { "arch": "rnn", "seq_len": 50, "final_val_acc": ..., "history": [...] }, ... ],
  "study_2": [ { "arch": "rnn", "critical_pos": 1, "input_grad_norm": 0.0, ... }, ... ],
  "study_3": [ { "arch": "rnn", "clip": 5.0, "final_val_acc": ..., ... }, ... ],
  "study_4": [ { "arch": "rnn", "depth": 2, "final_val_acc": ..., ... }, ... ]
}
```

`plot_results.py` reads this file and produces:

| File | Content |
|---|---|
| `figures/study1_sequence_length.pdf` | Accuracy + gradient norm vs T |
| `figures/study2_critical_position.pdf` | Input gradient norm + accuracy vs k |
| `figures/study3_clipping.pdf` | Accuracy + gradient norm vs clip threshold |
| `figures/study4_depth.pdf` | Accuracy vs number of layers |

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
