import argparse

import torch
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from datetime import datetime
import torch.nn.functional as F

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default='res.txt', help="Path for saving experimental results. Default is res.txt.")
    parser.add_argument("--task", type=str, default='DRL', help="Name of the task. Supported names are: DRL, random, semi-supervised, traditional. Default is DRL.")
    parser.add_argument("--layers", nargs='?', default='[256]', help="The number of units of each layer of the GNN. Default is [256]")
    parser.add_argument("--N_pred_hid", type=int, default=64, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--G_pred_hid", type=int, default=16, help="The number of hidden units of layer of the predictor. Default is 512")
    
    parser.add_argument("--eval_freq", type=float, default=5, help="The frequency of model evaluation")
    parser.add_argument("--mad", type=float, default=0.9, help="Moving Average Decay for Teacher Network")
    parser.add_argument("--Glr", type=float, default=0.0000006, help="learning rate")
    parser.add_argument("--Nlr", type=float, default=0.00001, help="learning rate")    
    parser.add_argument("--Ges", type=int, default=50, help="Early Stopping Criterion")
    parser.add_argument("--Nes", type=int, default=2000, help="Early Stopping Criterion")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--Gepochs", type=int, default=105)
    parser.add_argument("--Nepochs", type=int, default=3000)

    return parser.parse_known_args()

class EMA:  #Exponential Moving Average
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def get_task(strs):
    tasks = ["DRL","random","semi-supervised","traditional"]
    if len(strs) == 1:
        return "DRL"
    if ("--task" in strs) and len(strs) == 2:
        return "DRL"
    if ("--task" not in strs) or len(strs)!=3:
        return False
    elif strs[-1] not in tasks:
        return False
    else:
        return strs[-1]

def init_weights(m):  #Model parameter initialization
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1, z2):
    f = lambda x: torch.exp(x / 0.05)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def get_loss(h1, h2):
    l1 = semi_loss(h1, h2)
    l2 = semi_loss(h2, h1)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    #set require_grad
    for p in model.parameters():
        p.requires_grad = val

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','root','epochs','isAnneal','dropout','warmup_step','clus_num_iters']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args): 
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)

