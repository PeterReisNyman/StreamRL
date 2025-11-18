# TODO: should create an script that updates and runs on my cuda:

# ssh Peter@192.168.1.205
# cd C:\Users\peter\Documents\gpt
# .\.venv\Scripts\activate

# del /f train.py

# powershell -Command "@'


# '@ | Set-Content -Path train.py"

# python train.py


'''
after 15000 traning iterations with 

lr = 0.005 if i < 100000 else 0.05

train ds loss: 2.4773855209350586
dev ds loss: 2.4875779151916504

lr = 0.0005 if i < 10000 elif i < 3000 else 0.1
'''



import math
import torch 
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import torch.nn as nn

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
# MLP
n_emb = 384
context_len = 256 # num of input wrds
batch_size = 64
# GPT
num_heads = 4
head_size = 16

iter_steps = 1000

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
print(f'Using device: {device}')   

# --- not implamented
# n_head = 6
# n_layer = 6
# eval_interval = 5
# --- not implamented
try:
    ds = open('/Users/peternyman/Documents/GitHub/StreamRL/tiny-llm/traning_sets/input.txt', 'r', encoding='utf-8').read()
except:
    ds = open(r'C:\Users\peter\Documents\gpt\input.txt', 'r', encoding='utf-8').read()

chars = sorted(list(set(ds)))
vocab_size = len(chars)
print("vocab_size:", vocab_size)

# string -> index
stoi = {s: i for i, s in enumerate(chars)}
# index -> string
itos = {i: s for i, s in enumerate(chars)}

# text to list of int
enconder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])

print(enconder('I am an AI'))
print(decoder([21, 1, 39, 51, 1, 39, 52, 1, 13, 21]))

print(''.join(chars))

def get_batch(ds):
    X, Y = [], []
    
    ix = torch.randint(0, len(ds) - context_len -1, (batch_size,))
    
    for i in ix:
        X.append([stoi[c] for c in ds[i : i+context_len]])
        Y.append([stoi[c] for c in ds[i+1 : i+context_len+1]])
    return torch.tensor(X), torch.tensor(Y)


n1 = int(len(ds) * 0.9)

dsTr = ds[:n1]
dsDev = ds[n1:]

X, Y = get_batch(dsTr)


class Head(nn.Module):
    """Self-attention head"""
    def __init__(self, n_emb, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)

        tril = torch.tril(torch.ones(T, T, device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads in parallel"""
    def __init__(self, n_emb, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class GPT(nn.Module):
    """GPT language model"""
    def __init__(self, n_emb, num_heads, head_size, vocab_size):
        super().__init__()
        self.heads = MultiHeadAttention(n_emb, num_heads, head_size)

        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(context_len, n_emb)

        # Language model head to project to vocabulary size
        self.lm_head = nn.Linear(num_heads * head_size, vocab_size)

    def forward(self, X):
        B, T = X.shape

        tok_emb = self.token_embedding_table(X)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        X = tok_emb + pos_emb

        X = self.heads.forward(X)

        # Project to vocabulary size
        logits = self.lm_head(X)

        return logits

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]  # Crop to context length

            # Forward pass
            logits = self(idx_cond)  # (B, T, vocab_size)

            # Extract only the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# Only run training when this file is executed directly, not when imported
if __name__ == "__main__":
    gpt = GPT(n_emb, num_heads, head_size, vocab_size).to(device)
    print([p.nelement() for p in gpt.parameters()])
    print(f'Model moved to {device}')

    lre = torch.linspace(-3, -1, iter_steps)
    lrs = 10**lre

    lri = []
    lrei = []
    lossi = []
    steps = 0

    parameters = list(gpt.parameters())
    for i in range(iter_steps):
        # construct minibatch
        Xb, Yb = get_batch(dsTr) # batch X,Y
        Xb, Yb = Xb.to(device), Yb.to(device)
        # forward pass
        logits = gpt.forward(Xb)
        B, T, C = logits.shape # Batch size X context length X Embedding dimension

        loss = F.cross_entropy(logits.view(B*T, C),
                               Yb.view(B*T)) # loss function

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 300 == 0:
            print(f'{i:7d}/{iter_steps:7d}: {loss.item():.4f}')

        # tracking learning rate
        lri.append(lr)
        lrei.append(lre[i])
        lossi.append(loss.item())

    # Save model checkpoint
    checkpoint = {
        'model_state_dict': gpt.state_dict(),
        'hyperparameters': {
            'n_emb': n_emb,
            'context_len': context_len,
            'num_heads': num_heads,
            'head_size': head_size,
            'vocab_size': vocab_size,
            'batch_size': batch_size,
        },
        'training_stats': {
            'loss_history': lossi,
            'lr_history': lri,
            'steps': iter_steps,
            'final_loss': lossi[-1],
        },
        'vocab': {
            'stoi': stoi,
            'itos': itos,
            'chars': chars,
        }
    }

    checkpoint_path = 'gpt_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f'\nModel checkpoint saved to {checkpoint_path}')
    print(f'Final training loss: {lossi[-1]:.4f}')


    def show_loss(ds):
        Xb, Yb = get_batch(ds)
        Xb, Yb = Xb.to(device), Yb.to(device)

        # forward pass
        logits = gpt(Xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V),
                               Yb.view(B*T))
        return loss.item()

    print(f'train ds loss: {show_loss(dsTr)}')
    print(f'dev ds loss: {show_loss(dsDev)}')



