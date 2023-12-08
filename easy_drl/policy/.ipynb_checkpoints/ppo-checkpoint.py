import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from easy_drl.buffer.base_buffer import BaseBuffer
from easy_drl.network.actor_critic import ActorCritic
from easy_drl.utils import calculate_nature_cnn_out_dim


class Config:
    def __init__(self, input_type, input_size=None):
        self.max_buffer = 2048
        self.trainable_std = False
        self.use_cuda = True
        self.trans = ["state", "action", "reward", "done", "log_prob"]
        self.lr = 0.0003
        self.gamma = 0.99
        self.lambda_ = 0.95

        self.train_epoch = 80
        self.clip_ratio = 0.2
        self.critic_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        self.action_std_init = 0.6
        self.action_std_decay_rate = 0.05
        self.action_std_min = 0.1
        self.action_std_update_freq = 100

        self.actor_layer = [32, 32]
        self.critic_layer = [32, 32]
        if input_type == "vector":
            self.encoder = "mlp"
            self.encoder_layer = [64, 64]
            self.feature_dim = 32
        elif input_type == "image":
            self.encoder = "cnn"
            self.encoder_layer = [[input_size[0], 32, 8, 4],
                                  [32, 64, 4, 2],
                                  [64, 64, 3, 1]]
            size_h, size_w = calculate_nature_cnn_out_dim(input_size[1], input_size[2])
            self.feature_dim = [int(64 * size_h * size_w), 256]


class PPO:
    def __init__(self, state_dim, action_dim, continuous=True, input_type="vector", args=None):
        if args is None:
            self.args = Config(input_type, state_dim)
        self.buffer = BaseBuffer(self.args.trans, self.args.max_buffer)
        self.model = ActorCritic(state_dim[0], action_dim, self.args.actor_layer, self.args.critic_layer,
                                 self.args.encoder, self.args.encoder_layer, self.args.feature_dim, continuous,
                                 self.args.action_std_init)
        if self.args.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def select_action(self, state):
        state = torch.Tensor(state).float()
        if self.args.use_cuda:
            state = state.cuda()
        action, action_log_prob = self.model.act(state)
        return action.detach().cpu(), action_log_prob.detach().cpu()

    def add_buffer(self, transition):
        self.buffer.add(transition)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def learn(self):
        self.model.train()
        data, size = self.buffer.get_data()
        states = data["state"]
        actions = data["action"]
        rewards = data["reward"]
        dones = data["done"]
        old_log_probs = data["log_prob"]

        # Monte Carlo estimate of returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # trans into tensor and send to cpu/gpu
        states = torch.stack(states).float()
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        if self.args.use_cuda:
            states = states.cuda()  # [batch_size, state_dim]
            actions = actions.cuda()  # [batch_size]
            old_log_probs = old_log_probs.cuda()  # [batch_size]
            returns = returns.cuda()  # [batch_size]

        loss_list = []
        for e in range(self.args.train_epoch):
            # Evaluating old actions and values
            log_probs, values, dist_entropy = self.model.evaluate(states, actions)
            # print("shape check", log_probs.shape, values.shape, dist_entropy.shape)
            values = values.squeeze(-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = nn.MSELoss()(values, returns)
            entropy_bonus = -dist_entropy
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_bonus
            loss = loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        self.model.eval()
        self.buffer.clear()
        return np.mean(loss_list)
