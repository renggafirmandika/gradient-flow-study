# run_ablation.py
import torch
import torch.nn as nn
import json
from pathlib import Path

from tasks.selective_copy import make_dataloader
from models.rnn import VanillaRNN
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import TransformerClassifier
from utils.gradient_metrics import GradientTracker
from train import train_one_epoch, evaluate, get_input_gradient_at_position

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE  = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 128    # RNN/LSTM/GRU hidden size
NUM_HEADS   = 4      # Transformer
FF_DIM      = 256    # Transformer feedforward dim
NUM_CLASSES = 2
EPOCHS      = 20
BATCH_SIZE  = 128
LR          = 1e-3
TRAIN_SIZE  = 8000
VAL_SIZE    = 2000

# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(arch: str, seq_len: int) -> nn.Module:
    if arch == 'rnn':
        return VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'lstm':
        return LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'gru':
        return GRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'transformer':
        return TransformerClassifier(
            VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM,
            num_layers=1, max_seq_len=seq_len, num_classes=NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

# ── Single experiment ──────────────────────────────────────────────────────────

def run_experiment(arch: str, seq_len: int, critical_pos: int,
                   clip: float = None, depth: int = 1) -> dict:
    """
    Train one model under one configuration. Returns a result dict.

    arch:         one of 'rnn', 'lstm', 'gru', 'transformer'
    seq_len:      total sequence length T
    critical_pos: position of the label token (0-indexed)
    clip:         gradient clipping threshold. None = no clipping.
    depth:        number of stacked layers (Study 4)
    """
    print(f"\n[{arch.upper()}] T={seq_len} k={critical_pos} clip={clip} depth={depth}")

    # Data
    train_loader = make_dataloader(TRAIN_SIZE, seq_len, critical_pos, BATCH_SIZE)
    val_loader   = make_dataloader(VAL_SIZE,   seq_len, critical_pos, BATCH_SIZE)

    # Model — depth>1 handled via num_layers arg
    model = make_model(arch, seq_len).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tracker   = GradientTracker(model)

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, DEVICE,
            tracker=tracker, grad_clip=clip
        )
        val_metrics = evaluate(model, val_loader, DEVICE)

        history.append({
            'epoch':          epoch,
            'train_loss':     train_metrics['loss'],
            'train_acc':      train_metrics['accuracy'],
            'val_acc':        val_metrics['accuracy'],
            'mean_grad_norm': train_metrics.get('mean_grad_norm'),
            'vanishing_ratio':train_metrics.get('vanishing_ratio'),
            'max_grad_norm':  train_metrics.get('max_grad_norm'),
        })

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d} | "
                  f"val_acc={val_metrics['accuracy']:.3f} | "
                  f"mean_grad={train_metrics.get('mean_grad_norm', 0):.2e}")

    # After training: measure input gradient at critical position
    # Use one batch from val set
    sample_seq, sample_label = next(iter(val_loader))
    sample_seq   = sample_seq[:1].to(DEVICE)    # single sample
    sample_label = sample_label[:1].to(DEVICE)

    input_grad_norm = get_input_gradient_at_position(
        model, sample_seq, sample_label, critical_pos, DEVICE
    )

    tracker.remove_hooks()

    return {
        'arch':             arch,
        'seq_len':          seq_len,
        'critical_pos':     critical_pos,
        'clip':             clip,
        'depth':            depth,
        'final_val_acc':    history[-1]['val_acc'],
        'input_grad_norm':  input_grad_norm,   # ← key metric for Study 2
        'history':          history
    }

# ── Ablation sweeps ────────────────────────────────────────────────────────────

ARCHS = ['rnn', 'lstm', 'gru', 'transformer']

def study_1_sequence_length():
    """Vary T. Critical token fixed near end (k = T-2) so task is solvable.
    This isolates gradient degradation as the primary variable."""
    results = []
    for T in [50, 100, 200, 500, 1000]:
        for arch in ARCHS:
            r = run_experiment(arch, seq_len=T, critical_pos=T-2)
            results.append(r)
    return results

def study_2_critical_position():
    """Fix T=500. Move critical token from near-output to near-input.
    Directly measures gradient signal decay over distance."""
    results = []
    T = 500
    for k in [498, 400, 250, 100, 10, 1]:
        for arch in ARCHS:
            r = run_experiment(arch, seq_len=T, critical_pos=k)
            results.append(r)
    return results

def study_3_gradient_clipping():
    """Fix arch=rnn, T=500. Vary clip threshold to isolate exploding gradient."""
    results = []
    for clip in [None, 10.0, 5.0, 1.0, 0.1]:
        r = run_experiment('rnn', seq_len=500, critical_pos=250, clip=clip)
        results.append(r)
    return results

def study_4_depth():
    """Vary number of stacked layers. Deeper = more gradient degradation in RNNs."""
    results = []
    for depth in [1, 2, 4]:
        for arch in ARCHS:
            r = run_experiment(arch, seq_len=200, critical_pos=100, depth=depth)
            results.append(r)
    return results

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    all_results = {
        'study_1': study_1_sequence_length(),
        'study_2': study_2_critical_position(),
        'study_3': study_3_gradient_clipping(),
        'study_4': study_4_depth(),
    }

    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll studies complete. Results saved to results/all_results.json")

    # NEW RANDOM COMMENT
    