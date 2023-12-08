import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

from easy_drl.network.encoder import MLPEncoder, CNNEncoder


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_layer, critic_layer, encoder=None, encoder_layer=None,
                 feature_dim=None, continuous=False, std_init=0.6):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        if continuous:
            self.action_var = torch.full((action_dim,), std_init * std_init)

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

        if self.continuous:
            layers = [nn.Linear(input_dim, actor_layer[0]),
                      nn.ReLU(inplace=True)]
            for i in range(len(actor_layer) - 1):
                layers.append(nn.Linear(actor_layer[i], actor_layer[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(actor_layer[-1], action_dim))
            self.actor = nn.Sequential(*layers)

        else:
            layers = [nn.Linear(input_dim, actor_layer[0]),
                      nn.ReLU(inplace=True)]
            for i in range(len(actor_layer) - 1):
                layers.append(nn.Linear(actor_layer[i], actor_layer[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(actor_layer[-1], action_dim))
            layers.append(nn.Softmax(dim=-1))
            self.actor = nn.Sequential(*layers)

        layers = [nn.Linear(input_dim, critic_layer[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(critic_layer) - 1):
            layers.append(nn.Linear(critic_layer[i], critic_layer[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_layer[-1], 1))
        self.critic = nn.Sequential(*layers)
        self.__network_init()

    def forward(self):
        raise NotImplementedError

    def __network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def act(self, state):
        if self.encoder is not None:
            state = self.encoder(state)
        if self.continuous:
            mu = self.actor(state).cpu()
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(mu, cov_mat)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def evaluate(self, state, action):
        if self.encoder is not None:
            state = self.encoder(state)
        if self.continuous:
            mu = self.actor(state).cpu()
            action_var = self.action_var.expand_as(mu)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(mu, cov_mat)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(action.cpu())
        dist_entropy = dist.entropy()
        value = self.critic(state)
        return action_log_prob.cuda(), value, dist_entropy.cuda()


# state_dim = 4
# action_dim = 2
# actor_layer = [5, 6]
# critic_layer = [7, 8]
# encoder = "mlp"
# encoder_layer = [9, 10]
# feature_dim = 11
#
# model = ActorCritic(state_dim, action_dim, actor_layer, critic_layer, encoder, encoder_layer, feature_dim, continuous=False)
# model.eval()
# # print(model)
#
# states = torch.ones((1, 4))
# actions, action_log_prob = model.act(states)
# action_log_prob, value, dist_entropy = model.evaluate(states, actions)
# print(actions, actions.shape)
