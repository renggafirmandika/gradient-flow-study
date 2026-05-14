# run_study2.py — Study 2: Critical Token Position
# Run this independently. Saves results to results/all_results.json.

import torch
import json
from pathlib import Path

from tasks.selective_copy import make_dataloader
from models.rnn import VanillaRNN
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import TransformerClassifier
from utils.gradient_metrics import GradientTracker
from train import train_one_epoch, evaluate, get_input_gradient_at_position

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE  = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_HEADS   = 4
FF_DIM      = 256
NUM_CLASSES = 2
EPOCHS      = 20
BATCH_SIZE  = 128
LR          = 1e-3
TRAIN_SIZE  = 8000
VAL_SIZE    = 2000
SEEDS       = [42, 123, 456]

T = 500

print(f"Device: {DEVICE}")

def make_model(arch, seq_len):
    if arch == 'rnn':
        return VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'lstm':
        return LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'gru':
        return GRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    elif arch == 'transformer':
        return TransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM,
                                     num_layers=1, max_seq_len=seq_len,
                                     num_classes=NUM_CLASSES)

def run_experiment(arch, critical_pos, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"\n[{arch.upper()}] T={T} k={critical_pos} seed={seed}")
    train_loader = make_dataloader(TRAIN_SIZE, T, critical_pos, BATCH_SIZE)
    val_loader   = make_dataloader(VAL_SIZE,   T, critical_pos, BATCH_SIZE)

    model     = make_model(arch, T).to(DEVICE)
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
        'arch': arch, 'seq_len': T, 'critical_pos': critical_pos,
        'clip': None, 'depth': 1, 'seed': seed,
        'final_val_acc':   history[-1]['val_acc'],
        'input_grad_norm': input_grad_norm,
        'history':         history,
    }

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    results = []
    for k in [498, 400, 350, 300, 250, 200, 150, 100, 50, 25, 10, 1]:
        for arch in ['rnn', 'lstm', 'gru', 'transformer']:
            for seed in SEEDS:
                results.append(run_experiment(arch, critical_pos=k, seed=seed))

    out_path = Path('results/all_results.json')
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    all_results['study_2'] = results
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\nStudy 2 complete. Saved to results/all_results.json")
