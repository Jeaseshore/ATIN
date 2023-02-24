import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd

import numpy as np
import random
import os

def to_var(var):
    if torch.is_tensor(var):
        torch.nan_to_num_(var)
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var


def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)


def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return


def reverse_tensor(tensor_):
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = Variable(torch.LongTensor(indices), requires_grad=False)

    if torch.cuda.is_available():
        indices = indices.cuda()

    return tensor_.index_select(1, indices)


def reverse(ret):
    for key in ret:
        ret[key] = reverse_tensor(ret[key])

    return ret


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getF1scores(target, pred, thresholds=None):
    if thresholds.all() == None:
        thresholds = np.sort(pred)[::-1]
    f1scores = []
    accscores = []
    allTP = np.sum(target)
    for eachthreshold in thresholds:
        y_pred = (pred >= eachthreshold)
        right = (y_pred == target)
        TP = 0
        for (eachp, eachr) in zip(y_pred, right):
            if eachp and eachr:
                TP += 1
        predP = np.sum(y_pred)
        if predP == 0:
            P = 0
        else:
            P = TP / np.sum(y_pred)
        R = TP / allTP
        if (P + R) == 0:
            temp_y_F1 = 0
        else:
            temp_y_F1 = 2.0 * P * R / (P + R)
        acc = np.sum(right) / len(target)
        # f1scores.append(sklearn.metrics.f1_score(target, y_pred))
        f1scores.append(temp_y_F1)
        accscores.append(acc)

    return np.array(f1scores), np.array(accscores), thresholds

