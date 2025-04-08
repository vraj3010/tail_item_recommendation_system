import torch
from collections import defaultdict
import random

def train_test_split_per_user(indices, user_ids, test_size=0.2, device="cuda"):

    user_to_indices = defaultdict(list)

    # Collect all indices per user
    for idx, user_id in zip(indices, user_ids):
        user_to_indices[user_id.item()].append(idx.item())

    train_indices = []
    test_indices = []

    # Split indices for each user
    for user_id, user_indices in user_to_indices.items():
        random.shuffle(user_indices)
        split_point = int(len(user_indices) * (1 - test_size))
        train_indices.extend(user_indices[:split_point])
        test_indices.extend(user_indices[split_point:])

    # Convert to PyTorch tensors and move to GPU
    train_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_indices, dtype=torch.long, device=device)

    return train_idx, test_idx
