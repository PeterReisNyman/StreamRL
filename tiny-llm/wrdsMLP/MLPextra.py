# https://blog.ezyang.com/2019/05/pytorch-internals/

import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F

from itertools import islice
import random

import unicodedata

import os
os.system('clear')
os.system('clear')

interwrds = ' \n".,?!:;=@-'
trash = '€â»«˜œ™'

table = {ord(c) : None for c in trash}


def norm_text(text):
    spaces = [i for i, char in enumerate(text) if char in interwrds]
    spaces.insert(0,-1)
    spaces.append(len(text))

    return [text[spaces[i-1]+1:spaces[i]].lower().translate(table) for i in range(1,len(spaces)) if text[spaces[i-1]+1:spaces[i]] != '']

# load for inference
ckpt = torch.load("mlp_ckpt.pt", map_location="cpu")
C  = ckpt["C"].clone().requires_grad_(False)
W1 = ckpt["W1"].clone().requires_grad_(False)
b1 = ckpt["b1"].clone().requires_grad_(False)
W2 = ckpt["W2"].clone().requires_grad_(False)
b2 = ckpt["b2"].clone().requires_grad_(False)  # set True if resuming training
wtoi, itow = ckpt["wtoi"], ckpt["itow"]
wrds = ckpt["wrds"]
TOTAL_DIM, MAXWRDS, INTER_LAYER = ckpt["TOTAL_DIM"], ckpt["MAXWRDS"], ckpt["INTER_LAYER"]

parameters = [C,W1,b1,W2,b2]


"Generating"

out = ['<BOS>']

for i in range(100):

    if i < MAXWRDS:
        inp = ['<PAD>']*(MAXWRDS-i) + out[:i]
    else:
        inp = out[i-MAXWRDS:i]

    print(f"|{' '.join(inp)}|")
    X = []
    X.append([wtoi.get(t, wtoi['<PAD>']) for t in inp])
    X = torch.tensor(X, dtype=torch.long)

    # forward pass
    emb = C[X]
    # print(emb.shape[0],emb.shape[1],emb.shape[2])
    h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
    logits = h @ W2 + b2

    # softmax
    temperature = 1.0
    logits = logits / temperature

    logits = logits - logits.max(dim=1, keepdim=True).values
    probs = torch.softmax(logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True)

    # sample from last row of probs
    assert torch.isfinite(probs).all() and (probs.min() >= 0), "Invalid probabilities encountered"
    res = torch.multinomial(probs[-1], num_samples=1, replacement=True).item()
    

    out.append(wrds[res])
    
    # if wrds[res] == '<EOS>':
    #     break
