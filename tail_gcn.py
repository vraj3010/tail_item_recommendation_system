import torch
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from loss import *


def add_nodes(graph,head_item, batch_items, num_additional_nodes=3):
    """
    Augments the graph by adding new nodes based on head-tail item interactions.
    """
    edge_index = graph.edge_index.clone()
    num_nodes = graph.num_nodes
    new_edges = []
    k=1
    for tail_item in batch_items:
        # Find the head item with the maximum common interactions (users) with the tail_item
        similar_head_item, max_common_interactions = max(
            [(h, len(set(graph.edge_index[0][graph.edge_index[1] == h].numpy()) &
                     set(graph.edge_index[0][graph.edge_index[1] == tail_item].numpy())))
             for h in head_item if h != tail_item],
            key=lambda x: x[1],  # Sort by maximum common interactions
            default=(None, 0)  # Default if no valid head item is found
        )

        # print(f"The most similar head item to tail item {tail_item} is: {similar_head_item}")
        # print(f"Maximum common interactions (users) with tail item: {max_common_interactions}")

        head_interactions = list(set(graph.edge_index[0][graph.edge_index[1] == similar_head_item]))
        random.shuffle(head_interactions)
        l = min(len(head_interactions), num_additional_nodes)
        selected_interactions = head_interactions[:l]

        for interaction in selected_interactions:
            new_edges.append([tail_item, interaction])

    new_edges = torch.tensor(new_edges, dtype=torch.long).T
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    return Data(edge_index=edge_index, num_nodes=num_nodes)


def add_edges(graph,tail_items, probability_std=1.0):

    edge_index = graph.edge_index.clone()
    num_nodes = graph.num_nodes

    new_edges = []

    # in num_nodes i will pass tail items
    for node in tail_items:
        # Find potential new connections
        # Step 1: Find 1st-layer users (direct neighbors of the node)
        first_layer_users = set(edge_index[0][edge_index[1] == node].tolist())
        # print("first_layer: ", first_layer_users)
        # Step 2: Find 2nd-layer items (items connected to 1st-layer users)
        first_layer_users_tensor = torch.tensor(list(first_layer_users))
        mask_2nd_layer = torch.isin(edge_index[0], first_layer_users_tensor)
        second_layer_items = set(edge_index[1][mask_2nd_layer].tolist())
        # print("second_layer items: ", second_layer_items)
        # Step 3: Find 3rd-layer users (users connected to these 2nd-layer items)
        second_layer_items_tensor = torch.tensor(list(second_layer_items))
        mask_3rd_layer = torch.isin(edge_index[1], second_layer_items_tensor)
        third_layer_users = set(edge_index[0][mask_3rd_layer].tolist())
        # print("third_layer users: ", third_layer_users)
        # Potential nodes: users connected to 3rd-layer items but not in the 1st-layer
        potential_nodes = third_layer_users - first_layer_users
        # Use Gaussian probability to add edges
        if len(potential_nodes)>len(first_layer_users):
            potential_nodes=random.sample(list(potential_nodes),len(first_layer_users))
        for target in potential_nodes:
            # random.random()<
            # Larger standard deviation
            nll_loss = F.gaussian_nll_loss(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([probability_std]))
            # if random.random() < F.gaussian_nll_loss(torch.tensor([0.0]), torch.tensor([0.0]),
            #                                          torch.tensor([probability_std])):
            new_edges.append([node, target])

    # Convert new edges to tensor and update graph
    new_edges = torch.tensor(new_edges, dtype=torch.long).T
    # print(new_edges)
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    return Data(edge_index=edge_index,num_nodes=num_nodes)


def tail_items_self_supervised_training(model, optimizer, int_edges, head_items, tail_items, H=32, tau=0.1):

    total_loss_tail = 0
    random.shuffle(tail_items)
    batch_size = len(tail_items) // H
    for batch in range(H):

        start_idx = batch * batch_size
        end_idx = len(tail_items) if batch == H - 1 else (batch + 1) * batch_size
        batch_items = tail_items[start_idx:end_idx]

        graph = Data(edge_index=int_edges, num_nodes=max(int_edges.flatten()) + 1)
        graph2 = Data(edge_index=int_edges, num_nodes=max(int_edges.flatten()) + 1)

        graph = add_nodes(graph, head_items, batch_items)
        graph2 = add_edges(graph2,batch_items)
        model.train()

        all_emb1, all_emb2 = model(graph), model(graph2)
        emb1=all_emb1[batch_items]
        emb2=all_emb2[batch_items]
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)  # Normalize along feature dimension
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # print("emb1 stats: min =", torch.min(emb1), "max =", torch.max(emb1), "mean =", torch.mean(emb1))
        # print("emb2 stats: min =", torch.min(emb2), "max =", torch.max(emb2), "mean =", torch.mean(emb2))
        # print("Any NaN in emb1?", torch.isnan(emb1).sum().item(), "Any NaN in emb2?", torch.isnan(emb2).sum().item())

        loss_tail = head_loss(emb1, emb2, tau)
        total_loss_tail += loss_tail.item()
        optimizer.zero_grad()
        loss_tail.backward()
        optimizer.step()

    return total_loss_tail
