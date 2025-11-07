# https://blog.ezyang.com/2019/05/pytorch-internals/

import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F

import os
os.system('clear')
os.system('clear')

TOTAL_DIM = 30
MAXWCHARS = 100
INTER_LAYER = 500


# words = open('/Users/peternyman/Documents/GitHub/StreamRL/tiny-llm/names.txt', 'r', encoding='utf-8').read().splitlines()
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

C = torch.randn((len(chars),TOTAL_DIM))
W1 = torch.randn((MAXWCHARS*TOTAL_DIM,INTER_LAYER))
b1 = torch.rand((1,INTER_LAYER))
W2 = torch.randn((INTER_LAYER, len(chars)))
b2 = torch.randn((1, len(chars)))

parameters = [C,W1,b1,W2,b2]
# maxemize loss == N(C.ROW(XS)) --> YS 
"""
N(C.ROW(XS))
N(C.ROW(XS))
N(C.ROW(XS))
N(C.ROW(XS))
------------
YS
"""
# C.shape
"""
a; x x x
b; x x x
c; x x x
d; x x x
e; x x x
"""
# INPUT to OUTPUT
"""
L1 = (C.ROW(XS-3) C.ROW(XS-2) C.ROW(XS-1) C.ROW(XS)) @ L0 
logits = O @ L1 #log-counts
# soft-max beging
counts = logits.exp() # equivalent to N
probs = counts / counts.sum(1, keepdim=True)
# soft-max end
"""
# L0
"""
[30,???]

"""
# L1
"""
[30,MAXWCHARS] @ L0
"""
# logits
"""
L1 @ [???,len(char)]
"""





"Create the Traning set"
X, Y = [], []

for w in words:
    chs = ['@'] + list(w) + ['@']

    for i in range(1,len(chs)):
        if i < MAXWCHARS:
            inp = ('*'*(MAXWCHARS-i)) + ''.join(chs[:i])
        else:
            inp = ''.join(chs[i-MAXWCHARS:i])
        out = chs[i]
        
        # print(f'[{inp}] : |{out}|')

        X.append([stoi[i] for i in inp])
        Y.append(stoi[out])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

# emb @ W1 + b1
"""
print(emb.shape) # torch.Size([40, 10, 3]) batch * max wrds * indEmb 

print(W1.shape) # torch.Size([30, 100])
print(b1.shape) # torch.Size([100])

print(emb @ W1 + b1) # FAULT: mat1 and mat2 shapes cannot be multiplied (400x3 and 30x100)

#hence:

inp = []
for batchEmb in emb: 
    concatinate = []
    for wordEmb in batchEmb:
        for indEmb in wordEmb: 
            concatinate.append(indEmb)
    inp.append(torch.tensor(concatinate))

inp = torch.stack(inp)

print(f'the input should have shape of: {inp.shape}') # torch.Size([40, 30])
print(f'each batchEmb has {batchEmb.shape} word x indEmb') # torch.Size([10, 3])
print(f'each wordEmb has{wordEmb.shape} indEmb') # hastorch.Size([3])
"""

lre = torch.linspace(-3, 0, 30000)
lrs = 10**lre

lri = []
lossi = []

for p in parameters:
    p.requires_grad = True

for i in range(30000):
    # contruct minibatch
    ix = torch.randint(0, X.shape[0], ((32,)))

    # forward pass
    emb = C[X[ix]]
    # print(emb.shape[0],emb.shape[1],emb.shape[2])
    h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    if i > 20000:
        lr = 0.01778
    else:
        lr = 0.1778 

    # lr = lrs[i] 
    for p in parameters:
        p.data += -lr * p.grad 

    # tracking learning rate
    # lri.append(lr)
    # lossi.append(loss.item())


# forward pass
emb = C[X]
# print(emb.shape[0],emb.shape[1],emb.shape[2])
h = torch.tanh(emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(f'Loss: {loss.item()}')


# save
ckpt = {
    "C": C.detach(), "W1": W1.detach(), "b1": b1.detach(),
    "W2": W2.detach(), "b2": b2.detach(),
    "stoi": stoi, "itos": itos,
    "TOTAL_DIM": TOTAL_DIM, "MAXWCHARS": MAXWCHARS, "INTER_LAYER": INTER_LAYER,
}
torch.save(ckpt, "mlp_ckpt.pt")

# plt.plot(lri,lossi)
# plt.show()
