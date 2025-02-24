# gcn_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Define a simple GCN Model
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Function to compute a_same
def compute_a_same(model, int_edges, head_items, tail_items):

    # Get user and item embeddings
    user_embeddings = model.user_embedding.weight  # (num_users, embedding_dim)
    item_embeddings = model.item_embedding.weight  # (num_items, embedding_dim)

    # Concatenate user and item embeddings
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # (num_users + num_items, embedding_dim)

    # Extract Head and Tail embeddings
    R_head = all_embeddings[head_items]  # (num_head_items, embedding_dim)
    R_tail = all_embeddings[tail_items]  # (num_tail_items, embedding_dim)

    # Compute element-wise multiplication
    elementwise_mul = R_head * R_tail  # (num_tail_items, embedding_dim)

    # Define GCN model
    gcn = GCN(all_embeddings.shape[1], all_embeddings.shape[1])

    # Convert edges to long tensor format
    edge_index = int_edges.to(dtype=torch.long)

    # Pass through GCN
    a_same = gcn(elementwise_mul, edge_index)

    return a_same
