import torch
from torch import nn


class EdgeModel(nn.Module):
    def __init__(self, in_dim, e_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, e_dim)
        )

    def forward(self, e_k, v_rk, v_sk, u):
        return self.mlp(torch.cat([e_k, v_rk, v_sk, u]))


class NodeModel(nn.Module):
    def __init__(self, in_dim, v_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, v_dim)
        )

    def forward(self, e_i_agg, v_i, u):
        return self.mlp(torch.cat([e_i_agg, v_i.unsqueeze(0), u.unsqueeze(0)], dim=1))


class GlobalModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, e_agg, v_agg, u):
        return self.mlp(torch.cat([e_agg, v_agg, u.unsqueeze(0)], dim=1))
