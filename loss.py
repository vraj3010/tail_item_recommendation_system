import torch
from torch_geometric.utils import structured_negative_sampling
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def head_loss(emb1, emb2, tau):
    """
    Equation (3): Compute the head items' self-supervised loss.
    """
    sim_matrix = torch.matmul(emb1, emb2.T) / tau
    # print("sim_matrix min:", torch.min(sim_matrix).item(), "max:", torch.max(sim_matrix).item())
    eps = 1e-8
    exp_sim = torch.exp(sim_matrix)
    sum_exp_sim = torch.sum(exp_sim, dim=-1, keepdim=True)
    normalized_sim = exp_sim / (sum_exp_sim + 1e-8)
    diagonal_sim = torch.diagonal(normalized_sim)
    loss = -torch.sum(torch.log(diagonal_sim + 1e-8))
    return loss



