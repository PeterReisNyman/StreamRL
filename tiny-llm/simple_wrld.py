from noise import pnoise2
import numpy as np
from ursina import *
import math
import torch 
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import torch.nn as nn
                   #X   ,Y   ,Z
wrld = torch.zeros((1000,400,1000))


NATURE='1'

GROUND='2' 
STONE='3' 
ORE='4'

COAL='41'
IRON='42'
GOLD='43'

OAK='11'
MAPLE='12'
PINE='13' 
LINGONBERRY='14'

WOOD='1?1'
LEAF='1?2'
SAPLING='1?3'




def gerate_wrld():
    
    size = wrld.shape[0]
    layers = []
    for s in range(2, 4):
        layer = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                layer[y, x] = pnoise2(x / (10**s), (y / (10**s)), octaves=4)
        layers.append((s, layer))

    data = np.zeros((size, size))

    for s, layer in layers:
        weight = 2 ** s
        if s == 3:
            weight *= 10
        layer = np.where(layer < 0, layer * 0.1, layer)
        data += layer * weight            

    for y in data:
        


    plt.imshow(data, cmap="gray")
    plt.show()

gerate_wrld()