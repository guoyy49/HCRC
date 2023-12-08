import torch
from collections import deque


class BaseBuffer:
    def __init__(self, trans, max_len):
        self.trans = trans
        self.max_len = max_len
        self.data = {}
        for key in trans:
            self.data[key] = deque(maxlen=self.max_len)
        self.total_idx = 0

    def get_len(self):
        return self.total_idx

    def clear(self):
        """
        clear the buffer
        :return:
        """
        self.data = {}
        for key in self.trans:
            self.data[key] = []
        self.total_idx = 0

    def add(self, transition):
        """
        add a transition in buffer
        :return:
        """
        for key in transition:
            self.data[key].append(transition[key])
        self.total_idx += 1

    def get_data(self):
        data_size = len(self.data["state"])
        data = {}
        for key in self.data:
            data[key] = list(self.data[key])
        return data, data_size

    def sample(self, size):
        data_size = len(self.data["state"])
        size = min(data_size, size)
        indices = torch.randperm(data_size)[:size]
        data = {}
        for key in self.trans:
            data[key] = []
            for idx, ind in enumerate(indices):
                data[key].append(self.data[key][ind])
        return data

