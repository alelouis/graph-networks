import torch
from torch import nn

from graphnet import GraphNet


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, Nn):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.Nn = Nn
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 128), nn.ReLU(), nn.Linear(128, 4)
        )

    def forward(self, V):
        out = torch.empty((self.Nn, self.out_dim))
        for i in range(self.Nn):
            v = self.mlp(V[i])
            out[i] = v
        return out


class Classifier(nn.Module):
    def __init__(self, Ne, Nn, n_dim, e_dim):
        super().__init__()
        self.gn = GraphNet(Ne, Nn, n_dim, e_dim)
        self.dec = Decoder(n_dim, 4, Nn)

    def forward(self, E, V, u, r, s):
        E_prime, V_prime, u_prime = self.gn(E, V, u, r, s)
        z = self.dec(V_prime)
        return z
