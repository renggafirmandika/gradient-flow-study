# models/transformer.py
import torch
import torch.nn as nn
import math

class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.
    Uses mean pooling over all positions (not just the final one)
    to give it a fair chance on long sequences.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 ff_dim: int, num_layers: int, max_seq_len: int, num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_seq_len, embed_dim)  # learnable positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)

        embedded = self.embedding(x) + self.pos_encoding(positions)      # (batch, seq_len, embed_dim)
        encoded = self.transformer(embedded)                               # (batch, seq_len, embed_dim)

        pooled = encoded.mean(dim=1)          # mean pool over all positions
        return self.classifier(pooled)        # (batch, num_classes)