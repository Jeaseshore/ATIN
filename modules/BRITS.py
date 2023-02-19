import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from utils import reverse
from utils import reverse_tensor

import src.modules.mybrit as rits
from sklearn import metrics

from src.loss.lossFunction import py_sigmoid_focal_loss

SEQ_LEN = 48
RNN_HID_SIZE = 64

class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight=1, label_weight=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.rits_f = rits.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.rits_b = rits.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.re = nn.Sequential(
            nn.Linear(self.rnn_hid_size * 2,  self.rnn_hid_size),
            nn.BatchNorm1d(self.rnn_hid_size),
            nn.Sigmoid(),
            nn.Linear(self.rnn_hid_size, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        self.lossfunc = py_sigmoid_focal_loss

    def forward(self, data):
        data['forward']['sLabels'][:, -1] = 0
        data['backward']['sLabels'][:, -1] = 0
        data['backward'] = reverse(data['backward'])

        label = data['labels'].unsqueeze(-1)  # [batch_size,]
        label_ex = 1 - label
        label_ex = torch.concat((label, label_ex), dim=-1)

        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        h_f, h_b = ret_f['h_sub1'], ret_b['h_sub1']
        h = torch.cat([h_f, h_b], dim=1)
        pred = self.re(h)

        loss = self.lossfunc(pred, label_ex)
        loss_imputation = ret['loss']
        return {'lossImputation': loss_imputation, 'lossPred': loss, 'pred': pred, 'labels': label_ex}

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        # predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        # ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            loss = ret['lossPred'] + ret['lossImputation']
            loss.backward()
            optimizer.step()

        return ret

