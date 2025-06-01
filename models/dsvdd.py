from __future__ import absolute_import, print_function
import torch
from torch import nn


class DSVDD(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(DSVDD, self).__init__()
        
        step = (input_dim - latent_dim) / (num_layers + 1)
        
        layer_sizes = [input_dim]
        for i in range(num_layers-1):
            next_size = int(input_dim - (i + 1) * step)
            layer_sizes.append(next_size)
        layer_sizes.append(latent_dim)
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(in_features=layer_sizes[i], 
                                  out_features=layer_sizes[i + 1]))
            layers.append(nn.LeakyReLU())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.net(x)
        return output
