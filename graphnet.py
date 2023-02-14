import torch
from torch import nn

from aggregators import edge_to_node, edge_to_global, node_to_global
from models import EdgeModel, NodeModel, GlobalModel

"""
Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." 
arXiv preprint arXiv:1806.01261 (2018).
"""


class GraphNet(nn.Module):
    def __init__(self, Ne, Nn, n_dim, e_dim):
        super().__init__()
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.Nn = Nn
        self.Ne = Ne

        self.edge_model = EdgeModel(e_dim + 2 * n_dim + 1, self.e_dim)  # phi_e
        self.node_model = NodeModel(n_dim + 2, self.n_dim)  # phi_v
        self.global_model = GlobalModel(n_dim + e_dim + 1)  # phi_u
        self.edge_to_node_agg = edge_to_node  # rho_e_v
        self.edge_to_global_agg = edge_to_global  # rho_e_u
        self.node_to_global_agg = node_to_global  # rho_v_u

    def forward(self, E, V, u, r, s):
        E_prime = torch.empty((self.Ne, self.e_dim))
        for k in range(self.Ne):
            e_k, v_rk, v_sk = E[k], V[r[k]], V[s[k]]
            e_prime_k = self.edge_model(
                e_k, v_rk, v_sk, u
            )  # 1. Compute updated edge attributes
            E_prime[k] = e_prime_k

        V_prime = torch.empty((self.Nn, self.n_dim))
        for i in range(self.Nn):
            if any(r == i):
                E_prime_i = torch.stack(
                    [E_prime[k] for k in range(self.Ne) if r[k] == i], dim=0
                )
                e_prime_bar_i = self.edge_to_node_agg(
                    E_prime_i
                )  # 2. Aggregate edge attributes per node
                v_prime_i = self.node_model(
                    e_prime_bar_i, V[i], u
                )  # 3. Compute updated node attributes
                V_prime[i] = v_prime_i

        e_prime_bar = self.edge_to_global_agg(
            E_prime
        )  # 4. Aggregate edge attributes globally

        v_prime_bar = self.node_to_global_agg(
            V_prime
        )  # 5. Aggregate node attributes globally
        u_prime = self.global_model(
            e_prime_bar, v_prime_bar, u
        )  # 6. Compute updated global attribute

        return E_prime, V_prime, u_prime
