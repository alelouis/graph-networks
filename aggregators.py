import torch


def edge_to_node(E_i):
    e_agg_i = torch.sum(E_i, dim=0)
    return e_agg_i.unsqueeze(0)


def edge_to_global(E):
    e_agg = torch.sum(E, dim=0)
    return e_agg.unsqueeze(0)


def node_to_global(V):
    v_agg = torch.sum(V, dim=0)
    return v_agg.unsqueeze(0)
