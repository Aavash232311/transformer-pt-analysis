import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.load_pipeline import GenerateEvulatePairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' The hyperparamaters in which we got perfect accuracy '''
class FibonacciModDataset(Dataset):
    def __init__(self, seq_len=10, mod=10):
        self.mod = mod
        self.seq_len = seq_len

        self.global_seq = self.generate_fib_sequence(1000, mod)

        self.samples = []
        for i in range(0, len(self.global_seq) - seq_len - 1, seq_len):
            seq = self.global_seq[i : i + seq_len + 1]

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)

            self.samples.append((x, y))

    def generate_fib_sequence(self, length, mod):
        seq = [1, 1]
        for _ in range(length - 2):
            seq.append((seq[-1] + seq[-2]) % mod)
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def encode(k, N):
    k = k.float().to(device)
    return torch.stack([
        torch.sin(2 * math.pi * k / N),
        torch.cos(2 * math.pi * k / N)
    ], dim=-1)

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_layers=2, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.4)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.input_proj = nn.Linear(2, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)


        pos = torch.arange(T, device=tokens.device)

        x = encode(tokens, self.vocab_size)      
        x = self.input_proj(x)                 

        pos_emb = self.pos_embed(pos)         
        x = x + pos_emb.unsqueeze(0)        

        mask = torch.full((T, T), float('-inf'), device=tokens.device)
        for i in range(T):
            for j in range(max(0, i-2), i+1):
                mask[i, j] = 0.0


        for attn in self.layers:
            attn_out, _ = attn(x, x, x, attn_mask=mask)
            x = x + attn_out
        return self.out_proj(x)
    
    def get_embeddings(self):
        return self.pos_embed + self.token_embed
    


train_plot = []
eval_plot = []
def train_model(model, dataloader, test_loader, epochs=12, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})

        model.eval()
        with torch.no_grad():
            val_loss, total_acc = evaluate_model(model, test_loader, show_accuracy=False) 
            eval_plot.append(val_loss)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f}")

    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader, show_accuracy=False):
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            total_loss += loss.item()
            correct += (pred[:, 1:] == y[:, 1:]).sum().item()
            total += y[:, 1:].numel()

    total_accuracy = f"{correct / total:.2%}"
    if show_accuracy:
        print(f"Accuracy (eval mode): {total_accuracy}")

    avg_loss = total_loss / len(dataloader)
    total_accuracy = (correct / total) * 100
    return avg_loss, total_accuracy

if __name__ == "__main__":
    vocab_size = 53
    epoch = 256
    batch_size = 12
    total_accuray = 0
    generated_ds = FibonacciModDataset(mod=vocab_size, seq_len=2)
    eval_ds = GenerateEvulatePairs(generated_ds, mod=vocab_size)

    train_loader = DataLoader(generated_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,  num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = MinimalTransformer(vocab_size=vocab_size).to(device)


    ''' 
    We have made the model get high accuracy on limited possible resources, now we need to save the checkpoint in order to save time.
    '''


    checkpoint_dir = 'checkpoints'
    file_name = f'new_ds.pth'
    full_path  = os.path.join(checkpoint_dir, file_name)

    train_model(model=model, dataloader=train_loader, epochs=epoch, test_loader=test_loader) 
    avg_loss, eval_accuracy  = evaluate_model(model, test_loader, show_accuracy=True)



    if not os.path.exists(checkpoint_dir): # if this does not exists for some reason then create one
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_plot,
        'eval_loss_history': eval_plot,
        'epoch': epoch,
        'total_accuracy': total_accuray,
        'd_model': model.d_model
    }
    torch.save(checkpoint, full_path) 
    print(f"Successfully saved to: {full_path}")
