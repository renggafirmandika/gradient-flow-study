# run_study6.py — Study 6: Permuted MNIST Validation
#
# Validates that gradient flow failure patterns identified on the synthetic
# selective copy task generalise to real sequential data.
#
# Design: MNIST pixels fed one at a time (T=784). A fixed random permutation
# breaks local spatial structure, forcing genuine long-range dependency.
# All 4 architectures trained with the same hyperparameters as other studies.
# Input gradient norm measured at 5 representative positions to show decay.

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_HEADS   = 4
FF_DIM      = 256
NUM_CLASSES = 10        # MNIST digits 0–9
EPOCHS      = 20
BATCH_SIZE  = 128
LR          = 1e-3
SEEDS       = [42, 123, 456]

# Fixed permutation — same for every run so results are comparable across seeds
PERM_SEED   = 0
SEQ_LEN     = 784       # 28×28
GRAD_POSITIONS = [1, 100, 300, 500, 783]   # positions to measure input gradient

print(f"Device: {DEVICE}")

# ── Fixed permutation ─────────────────────────────────────────────────────────

perm = torch.randperm(SEQ_LEN, generator=torch.Generator().manual_seed(PERM_SEED))

def permute(x):
    return x[:, perm]

# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),           # flatten to (784,)
        transforms.Lambda(lambda x: x[perm]),              # apply fixed permutation
    ])
    train_ds = datasets.MNIST('data/', train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST('data/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

# ── Pixel models (linear projection instead of embedding) ────────────────────
# Each model projects each scalar pixel value to embed_dim via nn.Linear(1, embed_dim).
# This keeps the architecture identical to the token models except for the input layer.
# The attribute is named `projection` to distinguish it from nn.Embedding.

class RNNPixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, EMBED_DIM)
        self.rnn        = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        embedded = self.projection(x.unsqueeze(-1))   # (B, 784, embed_dim)
        _, h_n   = self.rnn(embedded)
        return self.classifier(h_n.squeeze(0))


class LSTMPixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, EMBED_DIM)
        self.lstm       = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        embedded      = self.projection(x.unsqueeze(-1))
        _, (h_n, _)   = self.lstm(embedded)
        return self.classifier(h_n.squeeze(0))


class GRUPixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, EMBED_DIM)
        self.gru        = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        embedded = self.projection(x.unsqueeze(-1))
        _, h_n   = self.gru(embedded)
        return self.classifier(h_n.squeeze(0))


class TransformerPixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection   = nn.Linear(1, EMBED_DIM)
        self.pos_encoding = nn.Embedding(SEQ_LEN, EMBED_DIM)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=0.1, batch_first=True
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier   = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        positions = torch.arange(SEQ_LEN, device=x.device).unsqueeze(0)
        embedded  = self.projection(x.unsqueeze(-1)) + self.pos_encoding(positions)
        encoded   = self.transformer(embedded)
        return self.classifier(encoded.mean(dim=1))


MODELS = {
    'rnn':         RNNPixel,
    'lstm':        LSTMPixel,
    'gru':         GRUPixel,
    'transformer': TransformerPixel,
}

# ── Gradient measurement at multiple positions ────────────────────────────────

def get_pixel_gradients_at_positions(model, sequence, label, positions, device):
    """
    Returns {pos: grad_norm} for each position in positions.
    sequence: (1, 784) float tensor — single sample
    Measures ∂loss/∂projection_output[pos] in one backward pass.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    x        = sequence.unsqueeze(-1)              # (1, 784, 1)
    embedded = model.projection(x)                 # (1, 784, embed_dim)
    embedded.retain_grad()

    if isinstance(model, (RNNPixel,)):
        _, h_n   = model.rnn(embedded)
        last_h   = h_n.squeeze(0)
    elif isinstance(model, LSTMPixel):
        _, (h_n, _) = model.lstm(embedded)
        last_h      = h_n.squeeze(0)
    elif isinstance(model, GRUPixel):
        _, h_n   = model.gru(embedded)
        last_h   = h_n.squeeze(0)
    elif isinstance(model, TransformerPixel):
        pos      = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
        embedded = embedded + model.pos_encoding(pos)
        embedded.retain_grad()
        encoded  = model.transformer(embedded)
        last_h   = encoded.mean(dim=1)

    logits = model.classifier(last_h)
    model.zero_grad()
    loss = criterion(logits, label)
    loss.backward()

    return {
        pos: embedded.grad[0, pos, :].norm().item()
        for pos in positions
        if embedded.grad is not None
    }

# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    grad_norms, vanishing = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()

        norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        if norms:
            grad_norms.append(np.mean(norms))
            vanishing.append(np.mean([n < 1e-4 for n in norms]))

        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return {
        'loss':           total_loss / len(loader),
        'accuracy':       correct / total,
        'mean_grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
        'vanishing_ratio':float(np.mean(vanishing))  if vanishing  else 0.0,
    }


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
    return {'loss': total_loss / len(loader), 'accuracy': correct / total}

# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(arch, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"\n[{arch.upper()}] Permuted MNIST T=784 seed={seed}")

    train_loader, val_loader = make_dataloaders()
    model     = MODELS[arch]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history   = []

    for epoch in range(1, EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, optimizer)
        val_m   = evaluate(model, val_loader)
        history.append({
            'epoch':           epoch,
            'train_loss':      train_m['loss'],
            'train_acc':       train_m['accuracy'],
            'val_acc':         val_m['accuracy'],
            'mean_grad_norm':  train_m['mean_grad_norm'],
            'vanishing_ratio': train_m['vanishing_ratio'],
        })
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d} | val_acc={val_m['accuracy']:.3f} | "
                  f"mean_grad={train_m['mean_grad_norm']:.2e}")

    # Measure input gradient at representative positions
    sample_x, sample_y = next(iter(val_loader))
    sample_x = sample_x[:1].to(DEVICE)
    sample_y = sample_y[:1].to(DEVICE)

    grad_at_positions = get_pixel_gradients_at_positions(
        model, sample_x, sample_y, GRAD_POSITIONS, DEVICE
    )

    return {
        'arch':              arch,
        'seq_len':           SEQ_LEN,
        'seed':              seed,
        'final_val_acc':     history[-1]['val_acc'],
        'grad_at_positions': grad_at_positions,   # {pos: grad_norm}
        'history':           history,
    }

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    results = []
    for arch in ['rnn', 'lstm', 'gru', 'transformer']:
        for seed in SEEDS:
            results.append(run_experiment(arch, seed=seed))

    out_path = Path('results/all_results.json')
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    all_results['study_6'] = results
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\nStudy 6 complete. Saved to results/all_results.json")
