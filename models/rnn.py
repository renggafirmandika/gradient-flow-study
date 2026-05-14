# models/rnn.py
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """
    Single-layer vanilla RNN for sequence classification.
    Uses the final hidden state to predict the label.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,      # input shape: (batch, seq_len, features)
            nonlinearity='tanh'
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) — token indices
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(embedded)       # output: (batch, seq_len, hidden_dim)
        last_hidden = h_n.squeeze(0)           # (batch, hidden_dim) — final timestep
        return self.classifier(last_hidden)    # (batch, num_classes)