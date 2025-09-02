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



def ndcg_calculation_2(model, test_set, neg_samples,num_users,head_items,k=5,N=None):

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
        if N2==None:
            N2=h
        neg_items = random.sample(neg_samples[user_id], N2)

        test_items = [item-num_users for item in pos_items] + neg_items

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


def ndcg_calculation_head(model, test_set, neg_samples, num_users, head_items, k=5,N=None):

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

        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    print(f"NDCG@{k} for head items: {avg_ndcg},{count}")
    return avg_ndcg


def ndcg_calculation_tail(model, test_set, neg_samples, num_users, tail_items, k=5,N=None):

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

            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k,len(head_pos_items))))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0

    print(f"NDCG@{k} for tail items: {avg_ndcg}")
    return avg_ndcg

def catalog_coverage_head_tail(model, test_set, num_users,neg_samples, head_items, tail_items, k=10, device="cpu"):
    user_embeddings = model.user_embedding.weight.to(device)
    item_embeddings = model.item_embedding.weight.to(device)
    num_items = item_embeddings.shape[0]

    recommended_items = set()
    recommended_head = set()
    recommended_tail = set()

    for user_id in range(num_users):
        # get user embedding
        user_emb = user_embeddings[user_id]
        # compute scores for all items
        neg_items=torch.tensor(neg_samples[user_id],dtype=torch.long,device=device)
        item_emb=item_embeddings[neg_items]
        scores = torch.matmul(item_emb, user_emb)
        # get top-k items
        topk_indices = torch.topk(scores, k).indices.tolist()
        # store recommended items
        for idx in topk_indices:
            recommended_items.add(idx)
            if idx in head_items:
                recommended_head.add(idx)
            if idx in tail_items:
                recommended_tail.add(idx)

    overall_coverage = len(recommended_items) / num_items
    head_coverage = len(recommended_head) / len(head_items) if len(head_items) > 0 else 0
    tail_coverage = len(recommended_tail) / len(tail_items) if len(tail_items) > 0 else 0

    print(f"Catalog Coverage@{k} (Overall): {overall_coverage:.4f}")
    print(f"Catalog Coverage@{k} (Head): {head_coverage:.4f}")
    print(f"Catalog Coverage@{k} (Tail): {tail_coverage:.4f}")

    return overall_coverage, head_coverage, tail_coverage

def precision_recall_at_k(model, test_set, neg_samples, num_users, head_items, k=10, N=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_embeddings = model.user_embedding.weight.to(device)
    item_embeddings = model.item_embedding.weight.to(device)

    total_precision, total_recall, count = 0, 0, 0
    head_precision, head_recall, head_count = 0, 0, 0
    tail_precision, tail_recall, tail_count = 0, 0, 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        h = len(pos_items)
        N2 = N if N is not None else h
        neg_items = random.sample(neg_samples[user_id], N2)

        # build candidate items
        test_items = [item - num_users for item in pos_items] + neg_items
        if len(test_items) < k:
            continue

        # move tensors to device
        test_items = torch.tensor(test_items, dtype=torch.long, device=device)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]

        # compute scores
        scores = torch.matmul(item_embs, user_emb)

        # get top-k indices
        _, topk_indices = torch.topk(scores, k)
        topk_items = test_items[topk_indices].detach().cpu().tolist()  # back to CPU

        # ground truth items for this user
        pos_set = set([item - num_users for item in pos_items])

        # hits
        hits = len(set(topk_items) & pos_set)
        precision = hits / k
        recall = hits / len(pos_set)

        # accumulate overall
        total_precision += precision
        total_recall += recall
        count += 1

        # head/tail split
        for item in pos_set:
            if item + num_users in head_items:
                head_precision += precision
                head_recall += recall
                head_count += 1
            else:
                tail_precision += precision
                tail_recall += recall
                tail_count += 1

    results = {
        "overall_precision": total_precision / count if count > 0 else 0,
        "overall_recall": total_recall / count if count > 0 else 0,
        "head_precision": head_precision / head_count if head_count > 0 else 0,
        "head_recall": head_recall / head_count if head_count > 0 else 0,
        "tail_precision": tail_precision / tail_count if tail_count > 0 else 0,
        "tail_recall": tail_recall / tail_count if tail_count > 0 else 0,
    }

    # 🔹 Print results in a clean format
    print("\n📊 Precision & Recall @", k)
    for key, val in results.items():
        print(f"{key}: {val:.4f}")

    return results
