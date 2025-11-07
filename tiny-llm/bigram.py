import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F

import os
os.system('clear')
os.system('clear')

# https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html

words = open('/Users/peternyman/Documents/GitHub/StreamRL/tiny-llm/names.txt', 'r', encoding='utf-8').read().splitlines()

# Build vocabulary and mappings
chars = sorted(list(set(''.join(words))))
chars.insert(0,'.')
# string -> index
stoi = {s: i for i, s in enumerate(chars)}
# index -> string
itos = {i: s for i, s in enumerate(chars)}
# Bigram count matrix sized by vocab
# N = torch.zeros(len(chars), len(chars), dtype=torch.int64)

"Create the Traning set"
xs = []
ys = []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for i in range(1,len(chs)):
        ch1 = chs[i-1]
        ch2 = chs[i]
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
num = xs.nelement()
ys = torch.tensor(ys)

xenc = F.one_hot(xs,num_classes=27).float()



W = torch.randn((27,27), requires_grad=True)
W.shape

"Traning"
for k in range(200):

    # And all of these are differentiable operations. So what we've done now is we are taking inputs, 
    # we have differentiable operations that we can backpropagate through, 
    # and we're getting out probability distributions.


    # forward pass
    logits = xenc @ W #log-counts
    # soft-max beging
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(1, keepdim=True)
    # soft-max end
    # loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # this is called regularazation which equals to P = (N+1).float() 
    loss = -probs[torch.arange(num), ys].log().mean() # Here we compare it with the YS
    print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad 





"Xai"
if False:
    print(f'x encoded is of shape {xenc.shape} # the input made into a one hot vector')
    print(f'Wheigts is of shape {W.shape}')
    print(f'logits is of shape {logits.shape}')
    print(f'counts is of shape {counts.shape}')
    print(f'probs is of shape {probs.shape}')
    print(f' W gives loss: {loss}')

# e.g
for let in range(0):
    print()
    [print(f'input: {chars[i]}') for i in range(len(xenc[let])) if xenc[let][i].item() == 1]
    print(f'Answer: {chars[ys[let]]}')
    pred_ix = probs[let].argmax().item()
    print(f'predicts: {chars[pred_ix]}')
    [print(f'|{char}| xenc:{xenc[let][i]:.0f}, probs:{probs[let][i]:.4f} ') for i, char in enumerate(chars)]




"Generating"
def gen(xs):
    xs = torch.tensor(xs)

    xenc = F.one_hot(xs,num_classes=27).float()

    # forward pass
    logits = xenc @ W #log-counts
    # soft-max beging
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(1, keepdim=True)
    # soft-max end
    return torch.multinomial(probs[-1], num_samples=1).item()


for _ in range(10):
    out = [0]
    while True:
        res = gen(out)
        out.append(res)
        
        if res == 0:
            break
    print("".join([chars[pred_ix] for pred_ix in out]))


"OLD Traning"
# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for i in range(1,len(chs)):
#         ch1 = chs[i-1]
#         ch2 = chs[i]
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]

#         N[ix1,ix2] += 1

# # P = (N+1).float() # this is called model smoothing 
# P = (N).float()
# P = P / P.sum(1, keepdim=True)



"OLD Generating"
# g = torch.Generator().manual_seed(9527148)

# for _ in range(10):
#     ix = 0

#     out = []
#     while True:
#         p = P[ix]
#         # p = N[ix].float()
#         # p = p / p.sum()
#         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         out.append(itos[ix])
#         if ix == 0:
#             break
#     print("".join(out[:-1]))



"Testing"
# likelihood
# log likelihood https://www.desmos.com/calculator (y = ln(x))
# loss function --> lower is better i.e negative log likelihood
# negative log likelihood

# log_likelihood = 0
# n = 0
# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for i in range(1,len(chs)):
#         ch1 = chs[i-1]
#         ch2 = chs[i]

#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]

#         prob = P[ix1,ix2]

#         log_prob = torch.log(prob)

#         log_likelihood += log_prob
#         n += 1
#         print(f'{ch1}{ch2} {prob:.4f}')

# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')