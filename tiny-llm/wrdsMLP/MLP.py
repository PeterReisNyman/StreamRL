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

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')


print(f'Using device: {device}')    

TOTAL_DIM = 30
MAXWRDS = 100
INTER_LAYER = 1000
N = 120000

interwrds = ' \n".,?!:;=@-'
trash = '€â»«˜œ™'

table = {ord(c) : None for c in trash}


def norm_text(text):
    spaces = [i for i, char in enumerate(text) if char in interwrds]
    spaces.insert(0,-1)
    spaces.append(len(text))

    return [text[spaces[i-1]+1:spaces[i]].lower().translate(table) for i in range(1,len(spaces)) if text[spaces[i-1]+1:spaces[i]] != '']

from datasets import load_dataset
# ds = load_dataset("roneneldan/TinyStories", split="train")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
print('Loaded Dataset')

wrd_set = []
for ex in islice(ds, N):
    wrd_set.extend(norm_text(ex['text']))  # flatten token lists
wrds = sorted(set(wrd_set))
# Ensure special tokens and keep them distinct
for tok in ['<PAD>', '<BOS>', '<EOS>']:
    if tok in wrds:
        wrds.remove(tok)
wrds.insert(0, '<PAD>')
wrds.insert(1, '<BOS>')
wrds.insert(2, '<EOS>')
print('Built token vocab')



# build mappings
wtoi = {w: i for i, w in enumerate(wrds)}
itow = {i: w for i, w in enumerate(wrds)}


C = torch.randn((len(wrds),TOTAL_DIM)).to(device)
W1 = torch.randn((MAXWRDS*TOTAL_DIM,INTER_LAYER)).to(device)
b1 = torch.randn((1,INTER_LAYER)).to(device)
W2 = torch.randn((INTER_LAYER, len(wrds))).to(device)
b2 = torch.randn((1, len(wrds))).to(device)

parameters = [C,W1,b1,W2,b2]



"traning"
def sample_batch(batch_size=32):
    Xb = torch.zeros((batch_size, MAXWRDS), dtype=torch.long).to(device)
    Yb = torch.zeros((batch_size,), dtype=torch.long).to(device)

    idxs = [random.randrange(N) for _ in range(batch_size)]

    for i in range(batch_size):
        # resample until we get enough tokens
        while True:
            idx = random.randrange(N)
            toks_core = norm_text(ds[idx]["text"])  # list of tokens
            if len(toks_core) >= 10:
                break

        toks = ['<BOS>'] + toks_core + ['<EOS>']

        # choose k with a right-tail bias, but enforce a minimum context length
        min_k = min(len(toks) - 1, max(10, MAXWRDS // 2))
        k = random.choices(range(min_k, len(toks)),
                           weights=[j*j for j in range(min_k, len(toks))],
                           k=1)[0]

        start = max(0, k - MAXWRDS)
        window = toks[start:k]

        # left-pad with <PAD> to fixed length
        if len(window) < MAXWRDS:
            window = ['<PAD>'] * (MAXWRDS - len(window)) + window

        Xb[i] = torch.tensor([wtoi.get(t, wtoi['<PAD>']) for t in window], dtype=torch.long).to(device)
        Yb[i] = wtoi.get(toks[k], wtoi['<PAD>'])
    return Xb, Yb

lre = torch.linspace(-3, 0, 10000)
lrs = 10**lre

lri = []
lossi = []

for p in parameters:
    p.requires_grad = True

for i in range(3000):
    Xb, Yb = sample_batch(32)

    # [print(f"-------------------\n{' '.join(itow[int(idx)] for idx in row)}") for row in Xb.detach().cpu().tolist()]

    # forward pass
    emb = C[Xb]
    # print(emb.shape[0],emb.shape[1],emb.shape[2])
    h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    if i > 2500:
        lr = 0.004
    else:
        lr = 0.04

    # lr = lrs[i] 
    for p in parameters:
        p.data += -lr * p.grad 

    # tracking learning rate
    lri.append(lr)
    print(f"---\n{lr}: {loss.item():4f}")
    lossi.append(loss.item())


# quick validation on a sampled batch
Xv, Yv = sample_batch(512)
Xv = Xv.to(device)
Yv = Yv.to(device)
emb = C[Xv]
h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yv)
print(f'Val loss: {loss.item()}')


# save
ckpt = {
    "C": C.detach(), "W1": W1.detach(), "b1": b1.detach(),
    "W2": W2.detach(), "b2": b2.detach(),
    "wtoi": wtoi, "itow": itow,
    "wrds": wrds,
    "TOTAL_DIM": TOTAL_DIM, "MAXWRDS": MAXWRDS, "INTER_LAYER": INTER_LAYER,
}
torch.save(ckpt, "mlp_ckpt.pt")

# plt.plot(lri,lossi)
# plt.show()
