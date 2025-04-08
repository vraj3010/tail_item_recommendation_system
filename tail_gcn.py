import torch
import torch.nn.functional as F
from loss import *
from torch_geometric.data import Data
import random


#deepseek version
def add_nodes(graph, head_item, batch_items, num_additional_nodes=3):
    device = graph.edge_index.device
    edge_index = graph.edge_index.clone()  # Shape: [2, num_edges]
    users, items = edge_index[0], edge_index[1]  # Assume edge_index = [users, items]
    new_edges = []

    # Precompute interactions for all head_items (efficiency)
    head_item_mask = (items.unsqueeze(0) == head_item.unsqueeze(1))  # [num_head, num_edges]

    for tail_item in batch_items:
        # Skip if tail_item already has edges (optional: reduces noise)
        tail_users = users[items == tail_item]
        if len(tail_users) > 0 and False:  # Set `True` to skip existing items
            continue

        # Find head_item with most common users (similarity)
        tail_mask = (items == tail_item).float().unsqueeze(0)  # [1, num_edges]
        common_counts = torch.mm(head_item_mask.float(), tail_mask.T).squeeze(1)  # [num_head]

        if not (common_counts > 0).any():  # Skip if no similarity
            continue

        # Get most similar head_item's users (excluding existing tail_users)
        similar_head_item = head_item[torch.argmax(common_counts)]
        candidate_users = users[(items == similar_head_item) & ~(users.unsqueeze(1) == tail_users.unsqueeze(0)).any(1)]

        if len(candidate_users) == 0:
            continue

        # Sample users (random or degree-weighted)
        if len(candidate_users) > num_additional_nodes:
            sampled_users = candidate_users[torch.randperm(len(candidate_users))[:num_additional_nodes]]
        else:
            sampled_users = candidate_users

        # Add edges in [user, item] format (critical for NDCG)
        new_edges.append(torch.stack([
            sampled_users,
            torch.full_like(sampled_users, tail_item)
        ], dim=0))

    # Merge new edges (if any)
    if new_edges:
        edge_index = torch.cat([edge_index, torch.cat(new_edges, dim=1)], dim=1)

    return Data(edge_index=edge_index, num_nodes=graph.num_nodes).to(device)

def add_edges(graph, tail_items):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = graph.edge_index.clone().to(device)
    num_nodes = graph.num_nodes

    # Initialize new_edges as an empty (2, 0) tensor
    new_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    for node in tail_items:

        first_layer_users = edge_index[0][edge_index[1] == node].unique()

        mask_2nd_layer = torch.isin(edge_index[0], first_layer_users)
        second_layer_items = edge_index[1][mask_2nd_layer].unique()

        mask_3rd_layer = torch.isin(edge_index[1], second_layer_items)
        third_layer_users = edge_index[0][mask_3rd_layer].unique()

        mask_not_in_first = ~torch.isin(third_layer_users, first_layer_users)
        potential_nodes = third_layer_users[mask_not_in_first]


        # Sample potential connections using tensor-based sampling
        num_first_layer = first_layer_users.numel()
        if potential_nodes.numel() > num_first_layer:
            sampled_indices = torch.randperm(potential_nodes.numel(), device=device)[:num_first_layer]
            potential_nodes = potential_nodes[sampled_indices]

        # Collect new edges
        if potential_nodes.numel() > 0:
            node_tensor = torch.full((1, potential_nodes.numel()), node, dtype=torch.long, device=device)
            new_edges_batch = torch.cat([potential_nodes.unsqueeze(0),node_tensor], dim=0)
            new_edges = torch.cat([new_edges, new_edges_batch], dim=1)



    # Concatenate new edges with the original graph
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    return Data(edge_index=edge_index, num_nodes=num_nodes).to(device)

def tail_items_self_supervised_training(model, optimizer, int_edges, head_items, tail_items, H=4, tau=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss_tail = 0

    int_edges = int_edges.to(device)
    model.to(device)

    tail_items_tensor = tail_items
    head_items_tensor = head_items

    shuffled_indices = torch.randperm(len(tail_items), device=device)
    tail_items_tensor = tail_items_tensor[shuffled_indices]

    batch_size = len(tail_items) // H
    base_graph = Data(edge_index=int_edges, num_nodes=max(int_edges.flatten()) + 1).to(device)

    ##########
    # monitor_edges = torch.tensor([[13,1025],[13,1095],[13,653],[13,1118]], dtype=torch.long, device=device)  # Shape: (3, 2)
    #
    # print("Selected interactions to monitor (node pairs):")
    # for i, (u, v) in enumerate(monitor_edges):
    #     print(f"Interaction {i + 1}: Node {u.item()} - Node {v.item()}")
    #
    # # Initial embedding product
    # graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)
    # with torch.no_grad():
    #     initial_emb = model(graph)
    #     print("\nInitial dot products between monitored interaction node pairs:")
    #     for i, (u, v) in enumerate(monitor_edges):
    #         dot_product = torch.dot(initial_emb[u], initial_emb[v]).item()
    #         print(f"Interaction {i + 1} ({u.item()}, {v.item()}): {dot_product:.4f}")
    # ##############
    graph = add_nodes(base_graph.clone(), head_items_tensor, tail_items)
    graph2 = add_edges(base_graph.clone(), tail_items)
    for batch in range(H):

        start_idx = batch * batch_size
        end_idx = len(tail_items) if batch == H - 1 else (batch + 1) * batch_size
        batch_items = tail_items_tensor[start_idx:end_idx]

        model.train()

        all_emb1, all_emb2 = model(graph), model(graph2)
        emb1 = all_emb1[batch_items]
        emb2 = all_emb2[batch_items]

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        loss_tail = head_loss(emb1, emb2, tau)
        total_loss_tail += loss_tail.item()

        optimizer.zero_grad()
        loss_tail.backward()
        optimizer.step()

        ####
        # with torch.no_grad():
        #     updated_emb = model(graph)
        #     print(f"\nDot products after batch {batch + 1}:")
        #     for i, (u, v) in enumerate(monitor_edges):
        #         dot_product = torch.dot(updated_emb[u], updated_emb[v]).item()
        #         print(f"Interaction {i + 1} ({u.item()}, {v.item()}): {dot_product:.4f}")
        #######

    return total_loss_tail
