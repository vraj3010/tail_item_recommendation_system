import numpy as np
import torch
import random
from torch_geometric.data import Data

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_interaction_edges(user_ids, movie_ids, ratings_, threshold=3.5):
    mask = ratings_ > threshold
    edges = np.stack([user_ids[mask], movie_ids[mask]])  # Include ratings
    return torch.LongTensor(edges).to(device)  # Move to GPU





def create_adj_matrix(int_edges, num_users, num_movies):
    n = num_users + num_movies
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move int_edges to device
    int_edges = int_edges.to(device)

    # Create an empty adjacency matrix on device
    adj = torch.zeros(n, n, device=device)

    # Create the interaction matrix (user x item)
    r_mat = torch.sparse_coo_tensor(
        int_edges,
        torch.ones(int_edges.shape[1], device=device),
        size=(num_users, num_movies)
    ).to_dense()

    # Fill upper-right and lower-left quadrants of bipartite adjacency
    adj[:num_users, num_users:] = r_mat.clone()
    adj[num_users:, :num_users] = r_mat.T.clone()

    # Convert to sparse COO format and return indices
    adj_coo = adj.to_sparse_coo().indices()

    return adj_coo


def separate_head_tail_items(interaction_counts, head_threshold=50):
    head_items = [item for item, count in interaction_counts.items() if count >= head_threshold]
    tail_items = [item for item, count in interaction_counts.items() if count < head_threshold]

    return torch.tensor(head_items, dtype=torch.long, device=device), \
        torch.tensor(tail_items, dtype=torch.long, device=device)

def get_negative_items(ratings):
    all_items = set(ratings['movieId'].unique())  # All available items
    user_interactions = ratings.groupby('userId')['movieId'].apply(set)  # Get interacted items per user

    negative_samples = {}  # Store negative samples for each user

    for user, interacted_items in user_interactions.items():
        negative_samples[user] = list(all_items - interacted_items)  # Items user has NOT interacted with

    return negative_samples  # Dictionary {userId: [negative_item1, negative_item2, ...]}


def create_test_set(test_edges):
    test_set = {}

    users = test_edges[0].tolist()  # Convert user tensor to list
    items = test_edges[1].tolist()  # Convert item tensor to list

    for user, item in zip(users, items):
        if user not in test_set:
            test_set[user] = []
        test_set[user].append(item)  # Store as tuple (item, rating)

    min_items = min(len(items) for items in test_set.values()) if test_set else 0
    print(f"Minimum number of items rated by any user: {min_items}")
    # print(len(test_set))
    return test_set



def ndcg_calculation_2(model, test_set, neg_samples,num_users,int_edges,head_items,k=5,N=None):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_embeddings=model.user_embedding.weight
    item_embeddings=model.item_embedding.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():

        if not pos_items:
            continue

        h=len(pos_items)
        N2=N
        # print(h,end=" ")
        if N2==None:
            N2=h
        # print(N2,end=" ")
        neg_items = random.sample(neg_samples[user_id], N2)

        # print(len(pos_items),end=" ")
        test_items = [item-num_users for item in pos_items] + neg_items
        # print(len(test_items))
        if len(test_items)<k:
            continue
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]
        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item+num_users in pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1/ np.log2(i + 2) for i in range(min(len(pos_items),k)))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg= total_ndcg / count if count > 0 else 0
    print(f"NDCG@10: {avg_ndcg}")


def ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=5,N=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    # with torch.no_grad():
    #     initial_emb = model(graph)
    #     user_embeddings = initial_emb[:num_users, :]
    #     item_embeddings = initial_emb[num_users:, :]
    user_embeddings=model.user_embedding.weight
    item_embeddings=model.item_embedding.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in head_items]
        if not head_pos_items:
            continue


        h = len(head_pos_items)
        N2=N
        if N2 is None:
            N2=h
        neg_items = random.sample(neg_samples[user_id], N2)



        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        if len(test_items)<k:
            continue
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # Check if the item is in the head positive items (convert back to original ID)
            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k,len(head_pos_items))))
        ndcg = dcg / idcg if idcg > 0 else 0

        # print(ndcg,user_id)
        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    print(f"NDCG@{k} for head items: {avg_ndcg},{count}")
    return avg_ndcg


def ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=5,N=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    # with torch.no_grad():
    #     initial_emb = model(graph)
    #     user_embeddings = initial_emb[:num_users, :]
    #     item_embeddings = initial_emb[num_users:, :]
    user_embeddings=model.user_embedding.weight
    item_embeddings=model.item_embedding.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in tail_items]
        if not head_pos_items:
            continue
        # print(len(head_pos_items),end=" ")

        h = len(head_pos_items)
        N2=N
        if N2 is None:
            N2=h
        neg_items = random.sample(neg_samples[user_id], N2)
        # if user_id==3:
        #     print(neg_items)
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        if len(test_items)<k:
            continue
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k,len(head_pos_items))))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0

    print(f"NDCG@{k} for tail items: {avg_ndcg}")
    return avg_ndcg
