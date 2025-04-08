import numpy as np
import torch
import random
from torch_geometric.data import Data

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_interaction_edges(user_ids, movie_ids, ratings_, threshold=3.5):
    mask = ratings_ > threshold
    edges = np.stack([user_ids[mask], movie_ids[mask], ratings_[mask]])  # Include ratings
    return torch.LongTensor(edges).to(device)  # Move to GPU


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
    ratings = test_edges[2].tolist()  # Convert rating tensor to list

    for user, item, rating in zip(users, items, ratings):
        if user not in test_set:
            test_set[user] = []
        test_set[user].append((item, rating))  # Store as tuple (item, rating)

    min_items = max(len(items) for items in test_set.values()) if test_set else 0
    # print(f"Minimum number of items rated by any user: {min_items}")

    return test_set



def ndcg_calculation_2(model, test_set, neg_samples,num_users,int_edges,head_items,k=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    with torch.no_grad():
        initial_emb = model(graph)
        user_embeddings = initial_emb[:num_users,:]
        item_embeddings = initial_emb[num_users:,:]

    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():

        if not pos_items:
            continue

        h=len(pos_items)


        neg_items = random.sample(neg_samples[user_id], h)

        if len(neg_items)*2<k:
            continue
        test_items = [item-num_users for item, _ in pos_items] + neg_items
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        ######
        # if user_id == 13:
        #     # Take the first `d` scores (assuming `d = len(pos_items)`)
        #     first_d_scores = scores[:len(pos_items)]  # shape: [d]
        #
        #     for item, score in zip(pos_items, first_d_scores):
        #         print(f"{item} (score: {score:.4f})", end=" | ")
        #     print()  # Newline after printing all items
        ###############


        sorted_indices = torch.argsort(scores, descending=True)

        # print("Test Items Length:", len(test_items))
        # print("Sorted Indices:", sorted_indices.tolist())
        # print("Test Items:", test_items)

        sorted_items = [test_items[i] for i in sorted_indices.tolist()]
        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # if(user_id==13):
            #     print(item+num_users,end="* ")

            if item+num_users in dict(pos_items):
                dcg += 1 / np.log2(i + 2)

        # if (user_id == 13):
        #     print()
        ideal_rels = sorted([r for _, r in pos_items], reverse=True)[:k]
        idcg = sum(1/ np.log2(i + 2) for i, _ in enumerate(ideal_rels))

        ndcg = dcg / idcg if idcg > 0 else 0
        # if(user_id<40):
        #     print(ndcg,end="&")
        #     print(user_id)
        total_ndcg += ndcg
        count += 1

    avg_ndcg= total_ndcg / count if count > 0 else 0
    print(f"NDCG@10: {avg_ndcg}")


def ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    with torch.no_grad():
        initial_emb = model(graph)
        user_embeddings = initial_emb[:num_users, :]
        item_embeddings = initial_emb[num_users:, :]

    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [(item, rating) for item, rating in pos_items if item in head_items]
        if not head_pos_items:
            continue

        h = len(head_pos_items)
        neg_items = random.sample(neg_samples[user_id], h)

        if len(neg_items) * 2 < k:
            continue

        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item, _ in head_pos_items] + neg_items
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # Check if the item is in the head positive items (convert back to original ID)
            if item + num_users in dict(head_pos_items):
                dcg += 1 / np.log2(i + 2)

        ideal_rels = sorted([r for _, r in head_pos_items], reverse=True)[:k]
        idcg = sum(1 / np.log2(i + 2) for i, _ in enumerate(ideal_rels))

        ndcg = dcg / idcg if idcg > 0 else 0
        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    # print(f"Number of users with head items: {count}")
    print(f"NDCG@{k} for head items: {avg_ndcg}")
    return avg_ndcg


def ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    with torch.no_grad():
        initial_emb = model(graph)
        user_embeddings = initial_emb[:num_users, :]
        item_embeddings = initial_emb[num_users:, :]

    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [(item, rating) for item, rating in pos_items if item in tail_items]
        if not head_pos_items:
            continue
        # print(len(head_pos_items),end=" ")
        if len(head_pos_items)*2 < k:
            continue
        h = len(head_pos_items)
        neg_items = random.sample(neg_samples[user_id], h)
        # if user_id==3:
        #     print(neg_items)
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item, _ in head_pos_items] + neg_items
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)
        #####
        # if user_id == 3:
        #     # Take the first `d` scores (assuming `d = len(pos_items)`)
        #     first_d_scores = scores[:h]  # shape: [d]
        #
        #     for item, score in zip(head_pos_items, first_d_scores):
        #         print(f"{item} (score: {score:.4f})", end=" | ")
        #     print()  # Newline after printing all items
        ##############
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # Check if the item is in the head positive items (convert back to original ID)
            # if(user_id==3):
            #     print(item+num_users,end="* ")
            if item + num_users in dict(head_pos_items):
                dcg += 1 / np.log2(i + 2)
        # if user_id==3:
        #     print()
        ideal_rels = sorted([r for _, r in head_pos_items], reverse=True)[:k]
        idcg = sum(1 / np.log2(i + 2) for i, _ in enumerate(ideal_rels))

        ndcg = dcg / idcg if idcg > 0 else 0
        # if user_id==3:
        #     print(ndcg,user_id)
        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    # print()
    # print(f"Number of users with tail items: {count}")
    print(f"NDCG@{k} for tail items: {avg_ndcg}")
    return avg_ndcg

