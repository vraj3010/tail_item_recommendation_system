import numpy as np
import torch
def create_interaction_edges(user_ids, movie_ids, ratings_, threshold=3.5):

    mask = ratings_ > threshold
    edges = np.stack([user_ids[mask], movie_ids[mask]])
    return torch.LongTensor(edges)


def separate_head_tail_items(interaction_counts, head_threshold=50):

    head_items = [item for item, count in interaction_counts.items() if count >= head_threshold]
    tail_items = [item for item, count in interaction_counts.items() if count < head_threshold]

    return head_items, tail_items


# def compute_a_same(model,int_edges, head_items, tail_items, num_users, num_movies):
#
#
#     # Step 1: Filter Edges to Keep Only Tail Items
#     is_tail_item = torch.isin(int_edges[1], torch.tensor(tail_items))  # Mask for tail items
#     tail_item_edges = int_edges[:, is_tail_item]  # Keep only edges with tail items
#
#     # Step 2: Convert to PyG Graph Format
#     tail_item_graph = Data(edge_index=tail_item_edges, num_nodes=num_users + num_movies)
#
#     # Step 3: Extract Embeddings
#     with torch.no_grad():
#         user_embeddings = model.user_embedding.weight  # (num_users, embedding_dim)
#         item_embeddings = model.item_embedding.weight  # (num_items, embedding_dim)
#         all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
#
#         R_head = all_embeddings[head_items]  # (num_head_items, embedding_dim)
#         R_tail = all_embeddings[tail_items]  # (num_tail_items, embedding_dim)
#
#     # Step 4: Compute Similarity Matrix
#     similarity_matrix = torch.matmul(R_tail, R_head.T)  # (num_tail_items, num_head_items)
#
#     # Step 5: Pass through GCN
#     gcn = LightGCN(num_users, num_movies, R_head.shape[1])  # Reuse LightGCN model
#     a_same = gcn(similarity_matrix)  # Output shape: (num_tail_items, embedding_dim)
#
#     return a_same


def get_matching_head_items(int_edges, head_items, tail_items):

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


def get_negative_items(ratings):

    negative_items = ratings[ratings['rating'] < 4].groupby('userId')['movieId'].apply(list).to_dict()

    # Find the user with the least number of negative items
    min_neg_items = min(len(items) for items in negative_items.values())

    print(f"Minimum number of negative items for any user: {min_neg_items}")
    return negative_items


def ndcg_at_k(true_relevance, predicted_scores, k=10):
    """Compute Normalized Discounted Cumulative Gain (NDCG@k)."""
    sorted_indices = torch.argsort(predicted_scores, descending=True)
    sorted_relevance = true_relevance[sorted_indices][:k]

    # Compute DCG
    dcg = (sorted_relevance / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))).sum()

    # Compute IDCG (Ideal DCG with perfect ranking)
    ideal_relevance = torch.sort(true_relevance, descending=True).values[:k]
    idcg = (ideal_relevance / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))).sum()

    return dcg / idcg if idcg > 0 else torch.tensor(0.0)  # Normalize