import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F

import os
os.system('clear')
os.system('clear')


words = open('/Users/peternyman/Documents/GitHub/StreamRL/tiny-llm/more.txt', 'r', encoding='utf-8').read().splitlines()

uppwrds = [w for w in words if w.isupper()]

dilogs = []
tmp = []

for w in words:
    if w in uppwrds:
        dilogs.append(' '.join(tmp))
        tmp = []
    elif w == '':
        pass
    else:
        tmp.append(w)

words = dilogs

# Build vocabulary and mappings
chars = sorted(list(set(''.join(words))))

chars.insert(0,'@')
chars.insert(1,'*') # Empty words
# print(chars)
# string -> index
stoi = {s: i for i, s in enumerate(chars)}
# index -> string
itos = {i: s for i, s in enumerate(chars)}

# load for inference
ckpt = torch.load("mlp_ckpt.pt", map_location="cpu")
C  = ckpt["C"].clone().requires_grad_(False)
W1 = ckpt["W1"].clone().requires_grad_(False)
b1 = ckpt["b1"].clone().requires_grad_(False)
W2 = ckpt["W2"].clone().requires_grad_(False)
b2 = ckpt["b2"].clone().requires_grad_(False)  # set True if resuming training
stoi, itos = ckpt["stoi"], ckpt["itos"]
TOTAL_DIM, MAXWCHARS, INTER_LAYER = ckpt["TOTAL_DIM"], ckpt["MAXWCHARS"], ckpt["INTER_LAYER"]

parameters = [C,W1,b1,W2,b2]

"Generating"

out = ['@']

for i in range(1000):

    if len(out) <= MAXWCHARS:
        inp = ('*'*(MAXWCHARS-len(out))) + ''.join(out)
    else:
        inp = ''.join(out[-MAXWCHARS:])


    # print(f'|{inp}|')
    X = []
    X.append([stoi[i] for i in inp])
    X = torch.tensor(X, dtype=torch.long)

    # forward pass
    emb = C[X]
    # print(emb.shape[0],emb.shape[1],emb.shape[2])
    h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
    logits = h @ W2 + b2

    # soft-max beging
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(1, keepdim=True)
    # soft-max end

    res = torch.multinomial(probs[-1], num_samples=1).item()

    out.append(chars[res])
    
    if res == 0:
        break

print(''.join(out))