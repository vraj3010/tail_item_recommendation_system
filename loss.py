import torch
from torch_geometric.utils import structured_negative_sampling
import torch.nn.functional as F

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


def tail_head_cosine_loss(head_emb, tail_emb, interaction_matrix, top_percent=0.1):

    num_head = head_emb.size(0)
    k = int(top_percent * num_head)

    topk_indices = torch.topk(interaction_matrix, k, dim=0).indices  # (k, num_tail)

    losses = []
    for j in range(tail_emb.size(0)):
        tail_vec = tail_emb[j].unsqueeze(0)  # (1, d)
        selected_heads = head_emb[topk_indices[:, j]]  # (k, d)

        sims = F.cosine_similarity(tail_vec, selected_heads, dim=-1)  # (k,)
        losses.append(-sims.mean())

    return torch.stack(losses).mean()