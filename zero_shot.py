import torch
import torch.nn as nn
from torch_geometric.data import Data

# Define device



class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, a_same, i_tail):
        input_vector = torch.cat([a_same, i_tail], dim=1)  # Eq (9)
        return self.model(input_vector)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


def synthetic_representation(generator, discriminator, a_same, i_head, i_tail, optimizer_G, optimizer_D, N_D):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    a_same, i_head, i_tail = a_same.to(device), i_head.to(device), i_tail.to(device)

    # Generate synthetic tail item representation
    i_G = generator(a_same, i_tail)

    # Compute discriminator scores
    score_G = discriminator(torch.cat([a_same, i_G], dim=1))
    score_head = discriminator(torch.cat([a_same, i_head], dim=1))

    # Discriminator loss
    loss_D = -score_G.mean() + score_head.mean()

    # Update Discriminator
    optimizer_D.zero_grad()
    loss_D.backward(retain_graph=True)  # Retain for Generator update
    optimizer_D.step()

    # Compute Generator loss
    loss_G = -discriminator(torch.cat([a_same, i_G.detach()], dim=1)).mean()

    # Update Generator
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    return loss_G.item(), loss_D.item()


def construct_interaction_graph(interaction_matrix, head_items, tail_items):

    # Convert interaction matrix to edge list
    edge_index = interaction_matrix.nonzero(as_tuple=True)  # Get indices of nonzero elements
    edge_weights = interaction_matrix[edge_index]  # Get corresponding weights

    # Map indices to actual item IDs
    edge_index = torch.stack([
        head_items[edge_index[0]],  # Map head indices to item IDs
        tail_items[edge_index[1]]  # Map tail indices to item IDs
    ], dim=0)

    # Construct graph
    # graph = Data(edge_index=edge_index, edge_weight=edge_weights)
    print(edge_weights.shape)
    print(edge_index.shape)
    return edge_index,edge_weights


def construct_interaction_matrix(int_edges, head_items, tail_items, num_users,num_movies):

    # Create a binary interaction matrix (users × items)
    interaction_matrix = torch.zeros((num_users, num_users + num_movies), device=int_edges.device)
    interaction_matrix[int_edges[0], int_edges[1]] = 1
    head=head_items-num_users
    tail=tail_items-num_users
    # Extract sub-matrices for head and tail items
    head_matrix = interaction_matrix[:, head_items]  # (num_users × num_head_items)
    tail_matrix = interaction_matrix[:, tail_items]  # (num_users × num_tail_items)

    # Compute common interactions between head and tail items
    interaction_matrix = head_matrix.T @ tail_matrix  # (num_head_items × num_tail_items)
    print(interaction_matrix.shape)
    return interaction_matrix
