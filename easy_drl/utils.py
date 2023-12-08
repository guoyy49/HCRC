import torch
import numpy as np


def make_transition(trans, *items):
    transition = {}
    for key, item in zip(trans, items):
        if isinstance(item, list):
            item = torch.stack(item)
            transition[key] = item
        elif isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
            transition[key] = item
        elif isinstance(item, torch.Tensor):
            transition[key] = item
        else:
            transition[key] = torch.Tensor([item])

    return transition


def make_batch(state, action, old_log_prob, advantage, old_value, learn_size, batch_size, use_cuda):
    batch = []
    total_indices = torch.randperm(learn_size)
    for i in range(learn_size // batch_size):
        indices = total_indices[batch_size * i: batch_size * (i + 1)]
        mini_state = torch.Tensor([])
        mini_action = torch.Tensor([])
        mini_old_log_prob = torch.Tensor([])
        mini_advantage = torch.Tensor([])
        mini_old_value = torch.Tensor([])
        if use_cuda:
            mini_state = mini_state.cuda()
            mini_action = mini_action.cuda()
            mini_old_log_prob = mini_old_log_prob.cuda()
            mini_advantage = mini_advantage.cuda()
            mini_old_value = mini_old_value.cuda()
        for ind in indices:
            mini_state = torch.cat((mini_state, state[ind].unsqueeze(0)), dim=0)
            mini_action = torch.cat((mini_action, action[ind].unsqueeze(0)), dim=0)
            mini_old_log_prob = torch.cat((mini_old_log_prob, old_log_prob[ind].unsqueeze(0)), dim=0)
            mini_advantage = torch.cat((mini_advantage, advantage[ind].unsqueeze(0)), dim=0)
            mini_old_value = torch.cat((mini_old_value, old_value[ind].unsqueeze(0)), dim=0)
        batch.append([mini_state, mini_action, mini_old_log_prob, mini_advantage, mini_old_value])
    return batch


def calculate_nature_cnn_out_dim(height, weight):
    size_h = np.floor((height - 8) / 4) + 1
    size_h = np.floor((size_h - 4) / 2) + 1
    size_h = np.floor((size_h - 3) / 1) + 1
    size_w = np.floor((weight - 8) / 4) + 1
    size_w = np.floor((size_w - 4) / 2) + 1
    size_w = np.floor((size_w - 3) / 1) + 1
    return size_h, size_w

