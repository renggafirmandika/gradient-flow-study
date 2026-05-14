# smoke_test.py
import torch
from tasks.selective_copy import make_dataloader
from models.rnn import VanillaRNN
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import TransformerClassifier
from utils.gradient_metrics import GradientTracker
from train import train_one_epoch, evaluate, get_input_gradient_at_position

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

VOCAB_SIZE  = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 128
SEQ_LEN     = 500
CRIT_POS    = 1     # near the end — easy task
BATCH_SIZE  = 32

models = {
    'rnn':         VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM),
    'lstm':        LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM),
    'gru':         GRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM),
    'transformer': TransformerClassifier(VOCAB_SIZE, EMBED_DIM, num_heads=4,
                       ff_dim=256, num_layers=1, max_seq_len=SEQ_LEN),
}

train_loader = make_dataloader(512, SEQ_LEN, CRIT_POS, BATCH_SIZE)
val_loader   = make_dataloader(128, SEQ_LEN, CRIT_POS, BATCH_SIZE)

for name, model in models.items():
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracker   = GradientTracker(model)

    # 3 epochs only
    for epoch in range(3):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        DEVICE, tracker=tracker, grad_clip=None)

    val_metrics = evaluate(model, val_loader, DEVICE)

    # Input gradient check
    sample_seq, sample_label = next(iter(val_loader))
    sample_seq   = sample_seq[:1].to(DEVICE)
    sample_label = sample_label[:1].to(DEVICE)
    input_grad   = get_input_gradient_at_position(model, sample_seq,
                                                  sample_label, CRIT_POS, DEVICE)

    tracker.remove_hooks()

    print(f"\n[{name.upper()}]")
    print(f"  val_acc:        {val_metrics['accuracy']:.3f}   (expect > 0.5, ideally > 0.7 after 3 epochs)")
    print(f"  mean_grad_norm: {train_metrics.get('mean_grad_norm', 0):.2e}  (expect non-zero)")
    print(f"  vanish_ratio:   {train_metrics.get('vanishing_ratio', 0):.3f}  (expect low at short seq)")
    print(f"  input_grad@k:   {input_grad:.2e}  (expect non-zero)")