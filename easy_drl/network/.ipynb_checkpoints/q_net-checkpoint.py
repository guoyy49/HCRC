import torch.nn as nn

from easy_drl.network.encoder import MLPEncoder, CNNEncoder


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, q_layer, encoder=None, encoder_layer=None, feature_dim=None):
        super(QNet, self).__init__()
        self.encoder = encoder
        if encoder is None:
            input_dim = state_dim
        elif encoder == "mlp":
            self.encoder = MLPEncoder(state_dim, encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        elif encoder == "cnn":
            self.encoder = CNNEncoder(encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, q_layer[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(q_layer)-1):
            layers.append(nn.Linear(q_layer[i], q_layer[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(q_layer[-1], action_dim))
        self.q_net = nn.Sequential(*layers)
        self.__network_init()

    def __network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, state):
        if self.encoder is not None:
            state = self.encoder(state)
        q = self.q_net(state)
        return q
