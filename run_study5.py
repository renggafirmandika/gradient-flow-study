# run_study5.py
# Study 5: Gradient Path Restoration via Attention
#
# Hypothesis: if gradient flow is the causal bottleneck, then augmenting a
# Vanilla RNN with attention (which restores an O(1) gradient path) should
# recover near-Transformer accuracy on the selective copy task.
#
# Design: fix T=500, vary k over the same positions as Study 2.
# Compare: Vanilla RNN | AttentionRNN | Transformer
# Expected: AttentionRNN matches Transformer; Vanilla RNN collapses.

import torch
import json
from pathlib import Path

from tasks.selective_copy import make_dataloader
from models.rnn import VanillaRNN
from models.attention_rnn import AttentionRNN
from models.transformer import TransformerClassifier
from utils.gradient_metrics import GradientTracker
from train import train_one_epoch, evaluate, get_input_gradient_at_position

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 10
EMBED_DIM  = 64
HIDDEN_DIM = 128
NUM_HEADS  = 4
FF_DIM     = 256
NUM_CLASSES = 2
EPOCHS     = 20
BATCH_SIZE = 128
LR         = 1e-3
TRAIN_SIZE = 8000
VAL_SIZE   = 2000

T = 500
POSITIONS = [498, 400, 250, 100, 10, 1]
ARCHS = ['rnn', 'attention_rnn', 'transformer']

def make_model(arch):
    if arch == 'rnn':
        return VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'attention_rnn':
        return AttentionRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'transformer':
        return TransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM,
                                     num_layers=1, max_seq_len=T,
                                     num_classes=NUM_CLASSES)
    raise ValueError(arch)

def run_experiment(arch, critical_pos):
    print(f"\n[{arch.upper()}] T={T} k={critical_pos}")

    train_loader = make_dataloader(TRAIN_SIZE, T, critical_pos, BATCH_SIZE)
    val_loader   = make_dataloader(VAL_SIZE,   T, critical_pos, BATCH_SIZE)

    model     = make_model(arch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tracker   = GradientTracker(model)
    history   = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, DEVICE,
                                        tracker=tracker, grad_clip=None)
        val_metrics   = evaluate(model, val_loader, DEVICE)
        history.append({
            'epoch':           epoch,
            'train_loss':      train_metrics['loss'],
            'train_acc':       train_metrics['accuracy'],
            'val_acc':         val_metrics['accuracy'],
            'mean_grad_norm':  train_metrics.get('mean_grad_norm'),
            'vanishing_ratio': train_metrics.get('vanishing_ratio'),
            'max_grad_norm':   train_metrics.get('max_grad_norm'),
        })
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d} | val_acc={val_metrics['accuracy']:.3f} | "
                  f"mean_grad={train_metrics.get('mean_grad_norm', 0):.2e}")

    sample_seq, sample_label = next(iter(val_loader))
    input_grad_norm = get_input_gradient_at_position(
        model, sample_seq[:1].to(DEVICE), sample_label[:1].to(DEVICE),
        critical_pos, DEVICE
    )
    tracker.remove_hooks()

    return {
        'arch':            arch,
        'seq_len':         T,
        'critical_pos':    critical_pos,
        'final_val_acc':   history[-1]['val_acc'],
        'input_grad_norm': input_grad_norm,
        'history':         history,
    }

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    results = []
    for k in POSITIONS:
        for arch in ARCHS:
            results.append(run_experiment(arch, k))

    # Merge into existing results file if it exists, else save separately
    out_path = Path('results/all_results.json')
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results['study_5'] = results

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nStudy 5 complete. Results saved to results/all_results.json")
