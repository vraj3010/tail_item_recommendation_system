import torch


def head_loss(emb1, emb2, tau):
    """
    Equation (3): Compute the head items' self-supervised loss.
    """
    sim_matrix = torch.matmul(emb1, emb2.T) / tau
    # print("sim_matrix min:", torch.min(sim_matrix).item(), "max:", torch.max(sim_matrix).item())
    eps = 1e-8
    exp_sim = torch.exp(sim_matrix)
    loss = -torch.sum(torch.log(exp_sim / (torch.sum(exp_sim, dim=-1, keepdim=True)+eps)))
    return loss






