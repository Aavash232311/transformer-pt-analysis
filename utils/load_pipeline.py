import torch
from torch.utils.data import Dataset, DataLoader



''' 
    Use this if you have the pairs, to load it into a dataset

'''
class AdhocTestDataset(Dataset):
    def __init__(self, sequences):
        self.samples = sequences  
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

''' 
    From x, y torch array for input output to data loader object.
'''

def get_dataset(seq, batch_size, shuffle): # leave this fasle in eval.
    ds = AdhocTestDataset(seq)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_sequence(a, b, mod=10, seq_len=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = [a, b] 

    while len(seq) < seq_len + 1:
        seq.append((seq[-1] + seq[-2]) % mod)

    # problem with this slicing. 
    x = torch.tensor(seq[:-1], dtype=torch.long).to(device=device)
    y = torch.tensor(seq[1:], dtype=torch.long).to(device=device)

    return x, y


''' 
    from (a, b) array pairs retruns x, and y
'''


def pairs(arr, mod, seq_len):
    res = []
    for a, b in arr:
        x, y = build_sequence(a, b, mod=mod, seq_len=seq_len)
        res.append((x, y))
    return res