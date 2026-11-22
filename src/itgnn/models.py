"""GNN models."""
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE node classifier (normal vs malicious)."""

    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x)
