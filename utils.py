import numpy as np
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from model import LightGCN
def create_interaction_edges(user_ids, movie_ids, ratings_, threshold=3.5):

    mask = ratings_ > threshold
    edges = np.stack([user_ids[mask], movie_ids[mask]])
    return torch.LongTensor(edges)


def separate_head_tail_items(interaction_counts, head_threshold=50):

    head_items = [item for item, count in interaction_counts.items() if count >= head_threshold]
    tail_items = [item for item, count in interaction_counts.items() if count < head_threshold]

    return head_items, tail_items


def compute_a_same(model,int_edges, head_items, tail_items, num_users, num_movies):
    """
    Compute A_same using the bipartite graph of item interactions.

    Parameters:
    - int_edges (Tensor): Interaction edges of shape [2, num_edges].
    - head_items (List[int]): List of head item indices.
    - tail_items (List[int]): List of tail item indices.
    - num_users (int): Number of users in the dataset.
    - num_movies (int): Number of items in the dataset.

    Returns:
    - Tensor: A_same matrix of shape [num_tail_items, embedding_dim].
    """

    # Step 1: Filter Edges to Keep Only Tail Items
    is_tail_item = torch.isin(int_edges[1], torch.tensor(tail_items))  # Mask for tail items
    tail_item_edges = int_edges[:, is_tail_item]  # Keep only edges with tail items

    # Step 2: Convert to PyG Graph Format
    tail_item_graph = Data(edge_index=tail_item_edges, num_nodes=num_users + num_movies)

    # Step 3: Extract Embeddings
    with torch.no_grad():
        user_embeddings = model.user_embedding.weight  # (num_users, embedding_dim)
        item_embeddings = model.item_embedding.weight  # (num_items, embedding_dim)
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        R_head = all_embeddings[head_items]  # (num_head_items, embedding_dim)
        R_tail = all_embeddings[tail_items]  # (num_tail_items, embedding_dim)

    # Step 4: Compute Similarity Matrix
    similarity_matrix = torch.matmul(R_tail, R_head.T)  # (num_tail_items, num_head_items)

    # Step 5: Pass through GCN
    gcn = LightGCN(num_users, num_movies, R_head.shape[1])  # Reuse LightGCN model
    a_same = gcn(similarity_matrix)  # Output shape: (num_tail_items, embedding_dim)

    return a_same


import torch

def get_matching_head_items(int_edges, head_items, tail_items):
    """
    Computes the most common head item for each tail item based on user interactions.

    Args:
        int_edges (torch.Tensor): Interaction edges (2, num_edges) where
                                  row 0 = users, row 1 = interacted items.
        head_items (list): List of head item indices.
        tail_items (list): List of tail item indices.

    Returns:
        torch.Tensor: A tensor where each index corresponds to the most common head item for a given tail item.
    """
    # Convert head_items and tail_items to tensors
    head_items = torch.tensor(head_items, dtype=torch.long)
    tail_items = torch.tensor(tail_items, dtype=torch.long)

    # Identify interactions for head and tail items
    head_mask = torch.isin(int_edges[1], head_items)
    tail_mask = torch.isin(int_edges[1], tail_items)

    user_head_interactions = int_edges[:, head_mask]  # Users who interacted with head items
    user_tail_interactions = int_edges[:, tail_mask]  # Users who interacted with tail items

    # Dictionary to store interaction counts
    interaction_counts = {}

    for user, tail_item in zip(user_tail_interactions[0], user_tail_interactions[1]):
        # Find head items this user interacted with
        user_head_items = user_head_interactions[1][user_head_interactions[0] == user]

        for head_item in user_head_items:
            key = (tail_item.item(), head_item.item())
            interaction_counts[key] = interaction_counts.get(key, 0) + 1

    # Find the most common head item for each tail item
    most_common_head = {}

    for (tail_item, head_item), count in interaction_counts.items():
        if tail_item not in most_common_head or count > most_common_head[tail_item][1]:
            most_common_head[tail_item] = (head_item, count)

    # Create a list where each tail item is mapped to its most common head item
    matching_head_items = [
        most_common_head[tail][0] if tail in most_common_head else head_items[0].item()
        for tail in tail_items
    ]

    # Convert to tensor for efficient indexing
    return torch.tensor(matching_head_items, dtype=torch.long)  # Shape: (num_tail_items,)
