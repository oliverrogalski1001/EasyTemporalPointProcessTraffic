import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.model = Sequential('x, edge_index', [
            (GCNConv(in_channels, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, out_channels),
        ])
        
    def forward(self, x):
        return self.model(x)