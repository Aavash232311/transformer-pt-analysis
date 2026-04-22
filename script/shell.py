import os
import time
import torch
import random
import torch.nn as nn
from .analysis.analysis_dft import Analysis
from torch.utils.data import Dataset, DataLoader
from utils.load_pipeline import GenerateEvulatePairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' The hyperparamaters in which we got perfect accuracy '''
class FibonacciModDataset(Dataset):
    def __init__(self, seq_len=10, mod=10):
        self.mod = mod
        self.seq_len = seq_len
        self.global_seq = self.generate_fib_sequence(mod=mod)

        self.samples = []
        for i in range(0, len(self.global_seq) - seq_len - 1, seq_len):
            seq = self.global_seq[i : i + seq_len + 1]

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)

            self.samples.append((x, y))

    ''' 
        Small shample of data for training less harder to memorize, 
        Accidently I have it only 9 seen pairs, and little gorking was seen.
    '''
    def generate_fib_sequence(self, mod):
        random.seed(42) 
        all_pairs = [(a,b) for a in range(mod) for b in range(mod)]
        random.shuffle(all_pairs)
        

        train_pairs = all_pairs[:int(0.3 * len(all_pairs))]  # ex:- 231 pairs
        
        seq = []
        for a, b in train_pairs:
            s = [a, b]
            for _ in range(self.seq_len + 1):
                s.append((s[-1] + s[-2]) % mod)
            seq.extend(s)
        
        return seq

    # def generate_fib_sequence(self, length, mod):
    #     seq = [1, 1]
    #     for _ in range(length - 2):
    #         seq.append((seq[-1] + seq[-2]) % mod)
    #     return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

class MLP(nn.Module):
    def __init__(self, d_model, hidden_layer=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, d_model),
        )

    def forward(self, x):
        return self.net(x)
    

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=1, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.mlps = nn.ModuleList([
            MLP(d_model)
            for _ in range(num_layers)
        ])


        self.out_proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.input_proj = nn.Linear(2, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)              
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)    

        mask = torch.full((T, T), float('-inf'), device=tokens.device)
        for i in range(T):
            for j in range(max(0, i-2), i+1):
                mask[i, j] = 0.0

        for attn, mlp in zip(self.layers, self.mlps):
            attn_out, _ = attn(x, x, x, attn_mask=mask)
            x = x + attn_out  
            x = x + mlp(x)
            
        return self.out_proj(x)
    
    def get_embeddings(self):
        return self.pos_embed + self.token_embed
    


train_plot = []
eval_plot = []
epoch_masses = []

train_accuracy = []
test_accuracy = []

def train_model(model, dataloader, test_loader, epochs=12, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()

    analysis = Analysis(vocab_size=model.vocab_size) 

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        correct = 0
        total = 0
        batch_d_masses = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            ''' For training accuracy. '''
            pred = logits.argmax(dim=-1)
            correct += (pred[:, 1:] == y[:, 1:]).sum().item()
            total += y[:, 1:].numel()

            ''' Diagonal spectral mass '''
            d_mass = analysis.diagonal_sperectal_mass(preferred_logits=logits, x=x)
            batch_d_masses.append(d_mass)

        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})
        epoch_masses.append(sum(batch_d_masses) / len(batch_d_masses))

        model.eval()
        with torch.no_grad():
            ''' evulate_model returns the loss and accuracy per epoch. '''
            val_loss, eval_accuracy = evaluate_model(model, test_loader, show_accuracy=False) 
            ''' Here we have accuray per ecpoh so we can append that in train accuracy '''
            test_accuracy.append({"epoch": epoch, "test_accuracy": eval_accuracy}) # for testing accuracy.
            eval_plot.append(val_loss)

        model.train()
        avg_loss = total_loss / len(dataloader)


        train_accuracy.append((correct / total) * 100)

        if (epoch + 1) % 100 == 0:
            checkpoint_dir = 'checkpoints'
            file_name = f'new_ds.pth'
            full_path  = os.path.join(checkpoint_dir, file_name)
            if not os.path.exists(checkpoint_dir): 
                os.makedirs(checkpoint_dir)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'train_loss_history': train_plot,
                'eval_loss_history': eval_plot,
                'epoch': epoch,
                'd_model': model.d_model,
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy
            }
            torch.save(checkpoint, full_path) 
            print(f"Successfully saved to: {full_path}", "- " * 20, "Checkpoint saved")
            print("10 interval")

        current_wd = optimizer.param_groups[0]['weight_decay']
        # if avg_loss < 0.05 and current_wd == 0.0:  # let it memorize first.
        #     for g in optimizer.param_groups:
        #         g['weight_decay'] = 1.9
        #         g['lr'] = 0.015
        #     print("Switched to weight decay phase")

        # if epoch >= 500:
        #     for g in optimizer.param_groups:
        #         g['lr'] = 0.00001
        current_lr = optimizer.param_groups[0]['lr']

        
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f} Learning rate(eta): {current_lr:.10f} weight decay {current_wd}")


    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader, show_accuracy=False):
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
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



def execute():
    vocab_size = 123
    epoch = 1000
    batch_size = 28 # change this as soon as you change the mod/vocab_size to make it a full batch.
    total_accuray = 0
    generated_ds = FibonacciModDataset(mod=vocab_size, seq_len=20)
    eval_ds = GenerateEvulatePairs(generated_ds, mod=vocab_size)

    train_loader = DataLoader(generated_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
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
        'd_model': model.d_model,
        "d_mass": epoch_masses,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy

    }
    torch.save(checkpoint, full_path) 
    print(f"Successfully saved to: {full_path}")

if __name__ == "__main__":
    execute()
    
