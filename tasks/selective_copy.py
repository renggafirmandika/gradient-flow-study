# tasks/selective_copy.py
import torch
from torch.utils.data import Dataset, DataLoader

class SelectiveCopyDataset(Dataset):
    """
    Selective copy task.
    
    Sequence of length T. Token at position k is the label (0 or 1).
    All other tokens are noise drawn from {2, ..., vocab_size-1}.
    
    Goal: predict the value at position k.
    This directly tests long-range dependency — harder as k moves earlier.
    """
    def __init__(self, num_samples: int, seq_len: int, critical_pos: int, vocab_size: int = 10):
        assert critical_pos < seq_len, "critical_pos must be < seq_len"
        assert vocab_size >= 4, "need at least 4 tokens: 0, 1, and 2+ noise tokens"
        
        self.seq_len = seq_len
        self.critical_pos = critical_pos
        self.vocab_size = vocab_size
        
        # Build all sequences at once (fast)
        # Noise tokens: {2, ..., vocab_size-1}
        noise = torch.randint(2, vocab_size, (num_samples, seq_len))
        
        # Label for each sample: 0 or 1
        labels = torch.randint(0, 2, (num_samples,))
        
        # Insert label token at critical position
        noise[:, critical_pos] = labels
        
        self.sequences = noise       # shape: (N, T)
        self.labels = labels         # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def make_dataloader(num_samples, seq_len, critical_pos, batch_size=128, vocab_size=10):
    dataset = SelectiveCopyDataset(num_samples, seq_len, critical_pos, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)