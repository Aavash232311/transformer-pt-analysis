import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class FibonacciModDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=10, mod=10):
        self.samples = []
        for _ in range(num_samples):
            a, b = random.randint(0, mod-1), random.randint(0, mod-1)
            seq = [a, b]
            for _ in range(seq_len - 2):
                seq.append((seq[-1] + seq[-2]) % mod)
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Minimal decoder-only Transformer (no MLP)
import torch.nn as nn

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=2, num_layers=1, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)
        attn_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)
        for attn in self.layers:
            attn_out, _ = attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
        return self.out_proj(x)

# Training loop

def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Evaluation loop

def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"Accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    vocab_size = 10
    generated_ds = FibonacciModDataset(num_samples=5000, mod=vocab_size)

    # split this into train and eval
    train_size = int(0.8 * len(generated_ds)) # 80% to train
    test_size = len(generated_ds) - train_size # rest of the size

    train_ds, test_ds = random_split(generated_ds, [train_size, test_size]) 

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = MinimalTransformer(vocab_size=vocab_size)
    train_model(model, train_loader)
    evaluate_model(model, test_loader)