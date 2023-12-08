import numpy as np


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)

class Encoder(nn.Module):

    def __init__(self, layer_config):
        super().__init__()
        self.stacked_gnn = nn.ModuleList(
            [GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=None)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x