# run_study3.py — Study 3: Gradient Clipping
# Run this independently. Saves results to results/all_results.json.

import torch
import json
from pathlib import Path

from tasks.selective_copy import make_dataloader
from models.rnn import VanillaRNN
from utils.gradient_metrics import GradientTracker
from train import train_one_epoch, evaluate, get_input_gradient_at_position

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE  = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_CLASSES = 2
EPOCHS      = 20
BATCH_SIZE  = 128
LR          = 1e-3
TRAIN_SIZE  = 8000
VAL_SIZE    = 2000

T            = 500
CRITICAL_POS = 250

print(f"Device: {DEVICE}")

def run_experiment(clip):
    print(f"\n[RNN] T={T} k={CRITICAL_POS} clip={clip}")
    train_loader = make_dataloader(TRAIN_SIZE, T, CRITICAL_POS, BATCH_SIZE)
    val_loader   = make_dataloader(VAL_SIZE,   T, CRITICAL_POS, BATCH_SIZE)

    model     = VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tracker   = GradientTracker(model)
    history   = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, DEVICE,
                                        tracker=tracker, grad_clip=clip)
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
        CRITICAL_POS, DEVICE
    )
    tracker.remove_hooks()

    return {
        'arch': 'rnn', 'seq_len': T, 'critical_pos': CRITICAL_POS,
        'clip': clip, 'depth': 1,
        'final_val_acc':   history[-1]['val_acc'],
        'input_grad_norm': input_grad_norm,
        'history':         history,
    }

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    results = []
    for clip in [None, 10.0, 5.0, 1.0, 0.1]:
        results.append(run_experiment(clip))

    out_path = Path('results/all_results.json')
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    all_results['study_3'] = results
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\nStudy 3 complete. Saved to results/all_results.json")
