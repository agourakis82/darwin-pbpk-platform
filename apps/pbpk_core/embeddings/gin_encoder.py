"""
Graph Isomorphism Network (GIN) Encoder for Molecular Graphs
==============================================================

Implements a 3-layer GIN for generating 256-dimensional graph embeddings.

Author: Dr. Agourakis
Date: October 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder for molecular graphs.
    
    Architecture:
        - 3 GINConv layers with ReLU activations
        - Global mean pooling
        - Output: 256-dimensional graph embedding
    """
    
    def __init__(
        self,
        node_features: int = 9,
        edge_features: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3
    ):
        """
        Initialize GIN encoder.
        
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features (not used in basic GIN)
            hidden_dim: Hidden dimension for intermediate layers
            output_dim: Output embedding dimension
            num_layers: Number of GIN layers
        """
        super(GINEncoder, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GIN layers
        self.convs = nn.ModuleList()
        
        # First layer: node_features -> hidden_dim
        nn1 = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            nn_mid = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_mid))
        
        # Last layer: hidden_dim -> output_dim
        nn_last = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.convs.append(GINConv(nn_last))
        
        # Batch normalization (optional, for stability)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
    
    def forward(self, data):
        """
        Forward pass through GIN encoder.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]
        
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            
            # ReLU activation for all layers
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return x
    
    def __repr__(self):
        return (
            f"GINEncoder(\n"
            f"  node_features={self.node_features},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  num_layers={self.num_layers}\n"
            f")"
        )

