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


def generator_loss(D, G, a_same, tail_items):
    """
    Equation (11): Generator loss function.
    """
    D_G = D(G(a_same, tail_items))
    loss = -torch.mean(D_G)
    return loss


def discriminator_loss(D, a_same, generated_tail, generated_head):
    """
    Equation (12): Discriminator loss function.
    """
    D_generated_tail = D(a_same, generated_tail)
    D_generated_head = D(a_same, generated_head)
    loss = torch.mean(D_generated_tail) - torch.mean(D_generated_head)
    return loss


def main_loss(predictions_pos, predictions_neg):
    """
    Equation (14): Main loss function (preference score).
    """
    loss = -torch.sum(torch.log(torch.sigmoid(predictions_pos - predictions_neg)))
    return loss
