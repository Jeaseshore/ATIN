import torch
import torch.nn as nn

from torch.autograd import Variable
from utils import reverse
from utils import reverse_tensor

import src.modules.Ti_lstm as ti

SEQ_LEN = 36


class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight, label_weight):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.ig = 3
        if self.ig >= SEQ_LEN:
            print('ignore point error')
            exit(0)

        self.build()

    def build(self):
        self.ti_f = ti.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.ti_b = ti.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)

    def forward(self, data):
        # data['forward']['sLabels'][:, -1] = 0
        # data['backward']['sLabels'][:, -1] = 0

        ig = self.ig
        ret_f = self.ti_f(data, 'forward', ig)
        data['backward'] = reverse(data['backward'])
        ret_b = reverse(self.ti_b(data, 'backward', ig))

        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        ret_b['h_n'] = reverse_tensor(ret_b['h_n'])
        ret_b['imputations'] = reverse_tensor(ret_b['imputations'])

        if self.ig > 0:
            consistency_f = ret_f['imputations'][:, self.ig:-self.ig]
            consistency_b = ret_b['imputations'][:, self.ig:-self.ig]
        elif self.ig == 0:
            consistency_f = ret_f['imputations']
            consistency_b = ret_b['imputations']
        loss_c = self.get_consistency_loss(consistency_f, consistency_b)

        loss = loss_f + loss_b + loss_c

        # predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        # ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        ret_f['h_n'] = torch.cat([ret_f['h_n'], ret_b['h_n']], dim=-1)

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
