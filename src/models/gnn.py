"""Graph Neural Network for water network pressure prediction.

MeshGraphNet-style architecture:
    1. Node encoder: MLP on node features
    2. Message passing: EdgeConv layers (aggregates neighbor info via edges)
    3. Decoder: MLP to predict pressure at each node

Physics constraint: mass conservation loss (flow in = flow out at each junction).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class EdgeConvLayer(MessagePassing):
    """Single message-passing layer with edge features.

    Message: MLP(x_i || x_j || edge_attr)
    Aggregation: mean
    Update: MLP(x_i || aggregated_messages)
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class WaterNetworkGNN(nn.Module):
    """GNN for predicting nodal pressures in a water distribution network.

    Architecture:
        Input node features → Encoder MLP → N message-passing layers → Decoder MLP → Pressure
    """

    def __init__(self, node_input_dim, edge_input_dim,
                 hidden_dim=128, n_layers=6, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conv_layers = nn.ModuleList([
            EdgeConvLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode
        x = self.encoder(x)
        edge_feat = self.edge_encoder(edge_attr)

        # Message passing with residual connections
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            x_new = conv(x, edge_index, edge_feat)
            x = norm(x + x_new)  # residual + layer norm
            x = self.dropout(x)

        # Decode to pressure
        pressure = self.decoder(x).squeeze(-1)
        return pressure
