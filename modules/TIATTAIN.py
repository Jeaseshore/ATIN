import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import src.modules.TI_net as ti_net
import src.modules.attain_seq_global as ATTAIN

from utils import reverse_tensor
import math
from src.loss.lossFunction import py_sigmoid_focal_loss

class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight=1, label_weight=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.build()

    def build(self):
        self.ti = ti_net.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.at = ATTAIN.Model(self.rnn_hid_size * 2 + self.input_size + 2, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.re = nn.Sequential(
            nn.Linear(self.rnn_hid_size,  self.rnn_hid_size),
            nn.BatchNorm1d(self.rnn_hid_size),
            nn.Sigmoid(),
            nn.Linear(self.rnn_hid_size, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

        # self.lossfunc = nn.CrossEntropyLoss()
        # self.lossfunc = nn.functional.binary_cross_entropy
        self.lossfunc = py_sigmoid_focal_loss

    def forward(self, data):
        data['forward']['sLabels'][:, -1] = 0
        data['backward']['sLabels'][:, -1] = 0

        label = data['labels'].unsqueeze(-1)              # [batch_size,]
        label_ex = 1 - label
        label_ex = torch.concat((label, label_ex), dim=-1)

        ret_ti = self.ti(data)    # [batch_size, seq, 2 * hidden]
        # data['forward']['values'] = ret_ti['h_n']
        ys = data['forward']['sLabels']
        data['forward']['values'] = torch.cat((ret_ti['h_n'], data['forward']['values'], ys), dim=2)
        # data['backward']['values'] = reverse_tensor(ret_ti['h_n'])

        ret_att = self.at(data, 'forward')
        h_f = ret_att[:, -1, :]
        pred = self.re(h_f)

        loss = self.lossfunc(pred, label_ex)

        loss_imputation = ret_ti['loss']
        return {'lossImputation': loss_imputation, 'lossPred': loss, 'pred': pred, 'labels': label_ex}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            loss = ret['lossPred'] + ret['lossImputation']
            loss.backward()
            optimizer.step()

        return ret
