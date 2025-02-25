import torch
import random
import torch.nn.functional as F
from loss import head_loss
from torch_geometric.data import Data


def head_items_self_supervised_training(model, optimizer, int_edges, head_items, tail_items, K=32,
                                        tau=0.1, deletion_rate=0.1):

    total_loss_head = 0
    random.shuffle(head_items)
    batch_size = len(head_items) // K
    for batch in range(K):

        start_idx = batch * batch_size
        end_idx = len(head_items) if batch == K - 1 else (batch + 1) * batch_size
        head_batch = head_items[start_idx:end_idx]
        graph=Data(edge_index=int_edges,num_nodes=max(int_edges.flatten())+1)
        model.train()
        all_emb1,all_emb2=model(graph,head_items=head_batch,mask_edges=True,mask_nodes=False),model(graph,head_items=head_batch,mask_edges=False,mask_nodes=True)

        emb1=all_emb1[head_batch]
        emb2=all_emb2[head_batch]

        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)  # Normalize along feature dimension
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        loss_head = head_loss(emb1, emb2, tau)
        total_loss_head += loss_head.item()

        optimizer.zero_grad()
        loss_head.backward()
        optimizer.step()

    return total_loss_head
