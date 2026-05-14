# train.py
import torch
import torch.nn as nn
from utils.gradient_metrics import GradientTracker


def _average_metrics(metrics_list):
    """Average gradient metrics across batches in an epoch."""
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0] if k != 'per_param']
    return {
        k: sum(m[k] for m in metrics_list if k in m) / len(metrics_list)
        for k in keys
    }


def train_one_epoch(model, dataloader, optimizer, device, tracker=None, grad_clip=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    epoch_grad_metrics = []

    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if tracker is not None:
            epoch_grad_metrics.append(tracker.get_metrics())
            tracker.reset()

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_metrics = _average_metrics(epoch_grad_metrics) if epoch_grad_metrics else {}
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        **avg_metrics
    }


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def get_input_gradient_at_position(model, sequence, label, critical_pos, device):
    """
    Returns the gradient norm of the loss w.r.t. the embedding
    at position critical_pos.

    Near zero = model completely ignored that position.

    sequence: (1, seq_len) — single sample, batch size 1
    label:    (1,)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    embedded = model.embedding(sequence)   # (1, seq_len, embed_dim)
    embedded.retain_grad()                 # keep gradient for this intermediate tensor

    # Run the rest of the forward pass past the embedding layer
    if hasattr(model, 'rnn'):
        output, h_n = model.rnn(embedded)
        last_hidden = h_n.squeeze(0)
    elif hasattr(model, 'lstm'):
        output, (h_n, c_n) = model.lstm(embedded)
        last_hidden = h_n.squeeze(0)
    elif hasattr(model, 'gru'):
        output, h_n = model.gru(embedded)
        last_hidden = h_n.squeeze(0)
    elif hasattr(model, 'transformer'):
        seq_len = sequence.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        embedded = embedded + model.pos_encoding(positions)
        embedded.retain_grad()
        encoded = model.transformer(embedded)
        last_hidden = encoded.mean(dim=1)
    else:
        raise ValueError("Unrecognised model architecture")

    logits = model.classifier(last_hidden)

    model.zero_grad()
    loss = criterion(logits, label)
    loss.backward()

    # embedded.grad shape: (1, seq_len, embed_dim)
    grad_at_k = embedded.grad[0, critical_pos, :]   # (embed_dim,)
    return grad_at_k.norm().item()