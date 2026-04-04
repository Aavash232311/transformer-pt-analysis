import torch
from collections import OrderedDict


class Analysis:

    def __init__(self, model, skeleton: OrderedDict[str, float], vocab_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size

        self.logit_lattice = torch.zeros(vocab_size, vocab_size, vocab_size)

        self.model = model.to(self.device)
        self.model.load_state_dict(skeleton)




    def discrete_fourier_transform(self):
        self.model.eval()

        ''' 
            Here we are doing a forward pass for logits making sure we don't have randomness
            from for example dropout.
        '''
        with torch.no_grad():
            for a in range(self.vocab_size):
                for b in range(self.vocab_size):
                    input_ids = torch.tensor([[a, b]]).to(self.device)
                    logits = self.model(input_ids)  # [1, seq_len, n]
                    self.logit_lattice[a, b] = logits[0, -1, :]

        # ℓ̃(c|a,b) = ℓ(c|a,b) - (1/n) Σ ℓ(c'|a,b)
        self.logit_lattice = self.logit_lattice - self.logit_lattice.mean(dim=-1, keepdim=True)
