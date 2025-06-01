import torch
import numpy as np
from torch import nn
    
def svddloss(data, center):
    return torch.sum((torch.tensor(data).float() - torch.tensor(center).float()) ** 2).item()    

def chunkfeed(model, data, size, outdim):
    temp = np.empty([data.shape[0], outdim], dtype=float)
    output = torch.tensor(temp).float()
    for i in range(0, data.shape[0], size):
        c = i + size
        chunk = torch.tensor(data[i:c, :]).float()
        output[i:c, :] = model(chunk).detach()
    return output    