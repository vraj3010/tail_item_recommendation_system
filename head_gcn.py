import torch
import torch.nn.functional as F
from loss import head_loss
from torch_geometric.data import Data

def head_items_self_supervised_training(model, optimizer, int_edges, head_items, K=128, tau=0.1, deletion_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss_head = 0

    int_edges = int_edges.to(device)
    head_items_tensor = head_items
    model.to(device)

    shuffled_indices = torch.randperm(len(head_items), device=device)
    head_items_tensor = head_items_tensor[shuffled_indices]
    batch_size = max(1, len(head_items) // K)


    # graph = Data(edge_index=int_edges, num_nodes=int_edges.max().item() + 1).to(device)


    for batch in range(K):
        start_idx = batch * batch_size
        end_idx = len(head_items) if batch == K - 1 else (batch + 1) * batch_size

        head_batch = head_items_tensor[start_idx:end_idx]


        all_emb1 = model(int_edges, head_items=head_batch, mask_edges=True, mask_nodes=False)
        all_emb2 = model(int_edges, head_items=head_batch, mask_edges=False, mask_nodes=True)
        # user_emb_0=model.user_embedding.weight
        # item_emb_0=model.item_embedding.weight
        # nu=user_emb_0.shape[0]
        # user_emb=all_emb1[:nu,:]
        # print(user_emb.shape)
        # item_emb=all_emb1[nu:,:]

        emb1 = F.normalize(all_emb1[head_batch], p=2, dim=1)
        emb2 = F.normalize(all_emb2[head_batch], p=2, dim=1)
        #
        #
        loss_head = head_loss(emb1, emb2, tau)
        total_loss_head += loss_head.item()

        optimizer.zero_grad()
        loss_head.backward()
        # loss.backward()
        optimizer.step()

    return total_loss_head
