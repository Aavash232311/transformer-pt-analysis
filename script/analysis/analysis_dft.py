import torch
from collections import OrderedDict
from utils.explained_variance_ratio import heat_map_raw_lattice


class Analysis:

    def __init__(self, model, skeleton: OrderedDict[str, float], vocab_size, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size

        # logit_lattice[a, b, c]
        self.logit_lattice = torch.zeros(vocab_size, vocab_size, vocab_size)

        self.uncentered_lattice = torch.zeros(vocab_size, vocab_size, vocab_size) # copy to see it in heatmap

        self.model = model.to(self.device)
        self.model.load_state_dict(skeleton)
        self.data = data

    def discrete_fourier_transform(self):
        self.model.eval()

        ''' 
            Here we are doing a forward pass for logits making sure we don't have randomness
            from for example dropout.
        '''
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.data:
                x = x.to(self.device)
                logits = self.model(x)  

                for i in range(x.shape[0]):
                    a = x[i, -2].item()   # second to last token
                    b = x[i, -1].item()   # last token
                    self.logit_lattice[a, b] = logits[i, -1, :] 

                pred = logits[:, -1, :].argmax(dim=-1)     
                expected = (x[:, -2] + x[:, -1]) % self.vocab_size  
                correct += (pred == expected).sum().item()
                total += x.shape[0]

        print(f"correct: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"unique (a,b) pairs seen: {(self.logit_lattice.sum(dim=-1) != 0).sum().item()}")

        self.uncentered_lattice = self.logit_lattice
        # ℓ̃ (c|a,b) = ℓ(c|a,b) - (1/n) Σ ℓ(c'|a,b) centered logits
        self.logit_lattice = self.logit_lattice - self.logit_lattice.mean(dim=-1, keepdim=True)

        n = self.vocab_size

        self.fourier_lattice = torch.zeros(n, n, n, dtype=torch.complex64)

        for c in range(n):
            phi = self.logit_lattice[:, :, c]          # shape [n, n] one class slice
            self.fourier_lattice[:, :, c] = torch.fft.fft2(phi) / n   # eq. 17

        self.power_spectrum = self.fourier_lattice.abs() ** 2  # shape [n, n, n] eqn 16, basically squaring

        return self.power_spectrum
    
    def diagonal_sperectal_mass(self):
        self.discrete_fourier_transform()
        return self.power_spectrum
    

    def heat_map(self):
        heat_map_raw_lattice(self.uncentered_lattice, self.vocab_size) # it's centered a unchanged after that

        