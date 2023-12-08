import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from easy_drl.buffer.base_buffer import BaseBuffer
from easy_drl.network.q_net import QNet
from easy_drl.utils import calculate_nature_cnn_out_dim


class Config:
    def __init__(self, input_type, input_size=None):
        self.max_buffer = 100000
        self.update_freq = 200
        self.use_cuda = True
        self.trans = ["state", "action", "reward", "next_state", "done"]
        self.lr = 0.001
        self.tau = 0.005
        self.gamma = 0.99
        self.batch_size = 128
        self.max_grad_norm = 1
        self.epsilon_init = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.q_layer = [256, 256]
        if input_type == "vector":
            self.encoder = "mlp"
            self.encoder_layer = [512, 256]
            self.feature_dim = 256
        elif input_type == "image":
            self.encoder = "cnn"
            self.encoder_layer = [[input_size[0], 32, 8, 4],
                                  [32, 64, 4, 2],
                                  [64, 64, 3, 1]]
            size_h, size_w = calculate_nature_cnn_out_dim(input_size[1], input_size[2])
            self.feature_dim = [int(64 * size_h * size_w), 256]


class DQN:
    def __init__(self, state_dim, action_dim, input_type="vector", args=None):
        if args is None:
            self.args = Config(input_type, state_dim)
        self.action_dim = action_dim
        self.buffer = BaseBuffer(self.args.trans, self.args.max_buffer)
        self.policy_net = QNet(state_dim[0], action_dim, self.args.q_layer, self.args.encoder, self.args.encoder_layer,
                               self.args.feature_dim)
        self.target_net = QNet(state_dim[0], action_dim, self.args.q_layer, self.args.encoder, self.args.encoder_layer,
                               self.args.feature_dim)
        if self.args.use_cuda:
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        self.update_network()
        self.policy_net.eval()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.lr)

        self.epsilon = self.args.epsilon_init

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() > epsilon:
            state = torch.Tensor(state)
            if self.args.use_cuda:
                state = state.cuda()
            q_value = self.policy_net(state)
            return torch.argmax(q_value).cpu().unsqueeze(0).detach()
        else:
            return torch.Tensor([random.choice(np.arange(self.action_dim))]).type(torch.int64).detach()

    def add_buffer(self, transition):
        self.buffer.add(transition)

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.args.epsilon_decay, self.args.epsilon_min)

    def update_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, model_path):
        torch.save(self.policy_net.state_dict(), model_path)

    def learn(self, step):
        self.policy_net.train()
        data = self.buffer.sample(self.args.batch_size)
        states = torch.stack(data["state"])  # [batch_size, state_dim]
        actions = torch.stack(data["action"])  # [batch_size, 1]
        rewards = torch.stack(data["reward"])  # [batch_size, 1]
        next_states = torch.stack(data["next_state"])  # [batch_size, state_dim]
        dones = torch.stack(data["done"])  # [batch_size, 1]
        # print("shape check", states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        if self.args.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
        actions = actions.type(torch.int64)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # get q-values for all actions in current states
        predicted_q = self.policy_net(states)  # [batch_size, action_dim]
        # select q-values for chosen actions
        predicted_q_actions = torch.gather(predicted_q, -1, actions)  # [batch_size, 1]
        # compute q-values for all actions in next states
        predicted_next_q = self.target_net(next_states)  # [batch_size, action_dim]
        # compute V*(next_states) using predicted next q-values
        next_state_values, indexes = torch.max(predicted_next_q, dim=-1)  # [batch_size]
        next_state_values = next_state_values.unsqueeze(-1)  # [batch_size, 1]
        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_q_actions = rewards + self.args.gamma * next_state_values * (1 - dones)  # [batch_size, 1]
        loss = nn.SmoothL1Loss()(predicted_q_actions, target_q_actions.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_net.eval()
        self.epsilon_decay()

        if step % self.args.update_freq == 0:
            print("update target network")
            self.update_network()
        return loss.item()
