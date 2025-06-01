from __future__ import absolute_import, print_function
import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, input_dim, dsvdd_latent_dim, ae_latent_dim, dsvdd_num_layers, ae_extra_layers):
        super(AE, self).__init__()
        self.z = ae_latent_dim
        
        step1 = (input_dim - dsvdd_latent_dim) / (dsvdd_num_layers + 1)
        encoder_sizes1 = [input_dim]
        for i in range(dsvdd_num_layers):
            next_size = int(input_dim - (i + 1) * step1)
            encoder_sizes1.append(next_size)
        
        step2 = (dsvdd_latent_dim - ae_latent_dim) / (ae_extra_layers + 1)
        encoder_sizes2 = [dsvdd_latent_dim]
        for i in range(ae_extra_layers):
            next_size = int(dsvdd_latent_dim - (i + 1) * step2)
            encoder_sizes2.append(next_size)
        encoder_sizes2.append(ae_latent_dim)
        
        encoder_sizes = encoder_sizes1 + encoder_sizes2[1:]
        
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(nn.Linear(in_features=encoder_sizes[i],
                                          out_features=encoder_sizes[i + 1]))
            encoder_layers.append(nn.LeakyReLU())
        
        decoder_sizes = encoder_sizes[::-1]
        
        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(nn.Linear(in_features=decoder_sizes[i],
                                          out_features=decoder_sizes[i + 1]))
            decoder_layers.append(nn.LeakyReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        
    def forward(self, x):
        f = self.encoder(x)        
        output = self.decoder(f)
        return output
