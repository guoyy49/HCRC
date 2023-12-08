import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, state_dim, layer_dim: list, feature_dim: int):
        super(MLPEncoder, self).__init__()
        layers = [nn.Linear(state_dim, layer_dim[0]),
                  # nn.BatchNorm1d(layer_dim[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            # layers.append(nn.BatchNorm1d(layer_dim[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_dim[-1], feature_dim, bias=False))
        self.encoder = nn.Sequential(*layers)
        self.out_dim = feature_dim

    def forward(self, x):
        return self.encoder(x)

    def get_dim(self):
        return self.out_dim


class CNNEncoder(nn.Module):
    def __init__(self, layer_dim: list, feature_dim: list):
        super(CNNEncoder, self).__init__()
        layers = []
        for layer in layer_dim:
            layers.append(nn.Conv2d(layer[0], layer[1], layer[2], layer[3]))
            # layers.append(nn.BatchNorm2d(layer[1]))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Linear(feature_dim[0], feature_dim[1], bias=False)
        self.out_dim = feature_dim[1]

    def forward(self, x):
        f = self.encoder(x)
        f = f.reshape(f.shape[0], -1)
        f = self.projector(f)
        return f

    def get_dim(self):
        return self.out_dim

