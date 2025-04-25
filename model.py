import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index,edge_weight=None):
        #row stores outgoing node and correspponding value stores incoming node in col
        row, col = edge_index
        #next line calculates the degree of each node(total incoming edges)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_weight is not None:
            norm = norm * edge_weight  # Apply interaction weight
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, edge_dropout=0.2, node_dropout=0.1, device=None,layer=3):

        super(LightGCN, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.num_layers=layer
        # Move embeddings to GPU
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(self.device)

        # Graph convolution layers on GPU
        self.convs=nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(LightGCNConv())

    def forward(self, edge_index, head_items=None, mask_edges=False, mask_nodes=False,edge_weights=None):

        edge_index = edge_index.to(self.device)
        if edge_weights is not None:
            edge_weights=edge_weights.to(self.device)
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0).to(self.device)
        emb_lst=[emb]

        if mask_edges and head_items is not None and self.edge_dropout > 0:
            remaining_edges = []

            for head_item in head_items:
                mask = edge_index[1] == head_item
                head_item_edges = edge_index[:, mask]

                retained_edges, _ = dropout_edge(head_item_edges, p=self.edge_dropout, training=self.training)
                remaining_edges.append(retained_edges)

            if remaining_edges:
                remaining_edges = torch.cat(remaining_edges, dim=1)

            mask = torch.isin(edge_index[1], head_items)
            edge_index = edge_index[:, ~mask]

            if remaining_edges.numel() > 0:
                edge_index = torch.cat([edge_index, remaining_edges], dim=1)

        if mask_nodes and head_items is not None and self.node_dropout > 0:
            head_nodes = set(head_items.tolist())  # Convert to Python set
            mask = torch.ones(emb.size(0), device=self.device)

            for head_item in head_nodes:
                if torch.rand(1, device=self.device) < self.node_dropout:
                    mask[head_item] = 0

            emb = emb * mask.unsqueeze(1)  # Apply node dropout


        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if edge_weights is not None:
            edge_weights = torch.cat([edge_weights, edge_weights], dim=0)
        for lightgcn in self.convs:
            emb=lightgcn(emb,edge_index,edge_weights)
            emb_lst.append(emb)
        # embeddings = self.conv1(embeddings, edge_index)
        # embeddings = self.conv2(embeddings, edge_index)

        out_emb=torch.stack(emb_lst,dim=1)
        out_emb=torch.mean(out_emb,dim=1)
        return out_emb

    def get_embeddings(self, nodes):
        """Safe way to get embeddings without breaking computation graph"""
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0).to(self.device)
        return embeddings[nodes]