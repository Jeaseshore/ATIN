import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math

from src.utils import reverse_tensor


# SEQ_LEN = 48

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight=1, label_weight=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size, self.rnn_hid_size)
        self.atten_linear = nn.Linear(self.input_size, self.input_size)
        self.a_softmax = nn.Softmax(dim=-1)

        # self.regression = nn.Linear(self.rnn_hid_size, 35)
        # self.temp_decay = TemporalDecay(input_size=35, rnn_hid_size=self.rnn_hid_size)

        # self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct):
        values = data[direct]['values']
        Deltas = data[direct]['Deltas']  # [batch_size, seq_length, 1]
        SEQ_LEN = values.size()[1]

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))
        c_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))

        Deltas = torch.cumsum(Deltas, dim=-2)  # cumsum seq [batch_size, seq_length, 1]
        Deltas = reverse_tensor(Deltas)

        data_T = torch.transpose(values, -1, -2)  # [batch_size, feature_size, seq_length]
        values = self.atten_linear(values)
        a_ij = torch.bmm(values, data_T)    # [batch_size, seq_length, seq_length]

        if torch.cuda.is_available():
            h, c, h_n, c_n = h.cuda(), c.cuda(), h_n.cuda(),  c_n.cuda()
        """
        inputs = values[:, 0, :]
        h, c = self.rnn_cell(inputs, (h, c))
        h_n[:, 0, :] = h
        c_n[:, 0, :] = c
        for t in range(1, SEQ_LEN, 1):
            x = values[:, t, :]         # [batch_size, input_size]
            a = a_ij[:, t, :t]          # [batch_size, t]
            d = Deltas[:, t, :t]        # [batch_size, t]
            alpha = self.a_softmax(a)   # [batch_size, t]
            alpha = alpha * 1 / torch.log(1 + d)  # d > 0
            c_mul_alpha = c_n[:, :t, :].clone() * alpha.unsqueeze(-1)   # [batch_size, t, hidden] * [batch_size, t, 1] = [batch_size, t, hidden]
                                                                        # c_n[:t].clone() is needed in loop
            C = torch.sum(c_mul_alpha, dim=-2)                          # [batch_size, t, hidden] -> [batch_size, hidden]
            h, c = self.rnn_cell(x, (h, C))
            h_n[:, t, :] = h
            c_n[:, t, :] = c

        """
        for t in range(0, SEQ_LEN - 1, 1):
            inputs = values[:, t, :]
            h, c = self.rnn_cell(inputs, (h, c))
            h_n[:, t, :] = h
            c_n[:, t, :] = c

        a_ij = self.a_softmax(a_ij[:, -1, :-1])  # [batch_size, seq_length - 1]
        a = a_ij.unsqueeze(-1)  # [batch_size, seq - 1, 1]
        d = Deltas[:, 1:]  # [batch_size, seq - 1, 1]
        c_i = a * c_n[:, :-1, :].clone() / torch.log(torch.e + d)  # [batch_size, seq - 1, hidden_size]/[batch_size, seq - 1, 1]
        C_i = torch.sum(c_i, dim=1)
        inputs = values[:, -1, :]
        h, c = self.rnn_cell(inputs, (h, C_i))
        h_n[:, -1, :] = h
        c_n[:, -1, :] = c

        return h_n


    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


if __name__ == '__main__':
    batch_seq = torch.randn(16, 8, 6).cuda()
    deltas = torch.ones(16, 8, 1).cuda()
    data = {'forward': {'values': batch_seq, 'Deltas': deltas}}
    m = Model(6, 32).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    h_n = m(data, 'forward')
    loss = torch.sum(h_n[:, -1, :] - h_n[:, -1, :] / 2)
    loss.backward()
    opt.step()
    pass
