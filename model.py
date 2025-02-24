import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, edge_dropout=0.2, node_dropout=0.1):
        """
        Initialize the LightGCN model with options for edge and node dropout.

        Parameters:
        - num_users: Number of users in the dataset.
        - num_items: Number of items in the dataset.
        - embedding_dim: Dimension of embeddings.
        - edge_dropout: Dropout rate for edges. If 0, no dropout will be applied.
        - node_dropout: Dropout rate for nodes. If 0, no dropout will be applied.
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.edge_dropout = edge_dropout  # Dropout rate for edges
        self.node_dropout = node_dropout  # Dropout rate for nodes

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Graph convolution layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

    def forward(self, graph, head_items=None, mask_edges=False, mask_nodes=False):
        """
        Forward pass of the model with optional dropout on edges and nodes.

        Parameters:
        - graph: PyG Data object containing edge_index (interaction graph).
        - head_items: List of head items for selective dropout (for edge dropout).
        - mask_edges: Boolean to enable/disable edge dropout.
        - mask_nodes: Boolean to enable/disable node dropout.
        """
        # Combine user and item embeddings into a single matrix
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        edge_index = graph.edge_index
        # print("graph.edge_index:", graph.edge_index)
        # **Edge Masking (Dropout on Edges for Head Items Only)**
        if mask_edges and head_items is not None and self.edge_dropout > 0:
            remaining_edges = []  # List to store edges that survive dropout

            for head_item in head_items:
                # Find edges where the target node is the current head item
                mask = edge_index[1] == head_item
                head_item_edges = edge_index[:, mask]  # Extract edges for this head item

                # Apply dropout: remove 10% of this head item's edges
                retained_edges, _ = dropout_edge(head_item_edges, p=self.edge_dropout, training=self.training)

                # Store the remaining edges
                remaining_edges.append(retained_edges)

            # Combine remaining edges from all head items
            if remaining_edges:
                remaining_edges = torch.cat(remaining_edges, dim=1)

            # Keep only edges that are not connected to head items
            mask = torch.isin(edge_index[1], torch.tensor(head_items, device=edge_index.device))
            edge_index = edge_index[:, ~mask]  # Remove all original head item edges

            # Add back only the remaining edges after dropout
            if remaining_edges.numel() > 0:
                edge_index = torch.cat([edge_index, remaining_edges], dim=1)

        # **Node Masking (Dropout on Nodes for Head Items Only)**
        if mask_nodes and head_items is not None and self.node_dropout > 0:
            # Step 1: Identify nodes connected to head items
            head_nodes = set(head_items)  # Make sure to include all head items
            mask = torch.ones(embeddings.size(0), device=embeddings.device)

            # Step 2: Randomly drop some nodes from head_items
            for head_item in head_nodes:
                if torch.rand(1) < self.node_dropout:
                    mask[head_item] = 0  # Drop the node

            # Step 3: Apply node dropout by masking out the corresponding embeddings
            embeddings = embeddings * mask.unsqueeze(1)  # Element-wise multiplication to mask embeddings

        # **Graph Convolution Layers**
        embeddings = self.conv1(embeddings, edge_index)
        embeddings = self.conv2(embeddings, edge_index)

        return embeddings


