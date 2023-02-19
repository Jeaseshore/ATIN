import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


# SEQ_LEN = 48

class CellStateT(nn.Module):
    def __init__(self, hidden_size):
        super(CellStateT, self).__init__()
        self.hidden_size = hidden_size

        self.build()

    def build(self):
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, c, D):
        c_s = self.linear(c)
        c_s = self.tanh(c_s)
        c_l = c - c_s
        c_l = c_l / torch.log(torch.e + D)
        c_hat = c_l + c_s

        return c_hat


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
    def __init__(self, input_size, rnn_hid_size, impute_weight, label_weight):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)

        self.regression = nn.Linear(self.rnn_hid_size, self.input_size)
        self.temp_decay = TemporalDecay(self.input_size, self.rnn_hid_size)
        self.c_decay = CellStateT(self.rnn_hid_size)

        # self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct, ig=4):
        # Original sequence
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        Deltas = data[direct]['Deltas']  # [batch_size, seq_len, 1]
        ys = data[direct]['sLabels']
        """
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
        """
        SEQ_LEN = values.size()[1]

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))

        if torch.cuda.is_available():
            h, c, h_n = h.cuda(), c.cuda(), h_n.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []
        pred_len = SEQ_LEN - ig
        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            D = Deltas[:, t]
            # y = ys[:, t, :]

            gamma = self.temp_decay(d)
            h = h * gamma

            x_h = self.regression(h)  # replace missing values with history data

            x_c = m * x + (1 - m) * x_h  # Imputation

            # x_c = torch.cat([x_c, y], dim=1)  # cat y

            # x_c_self = x_c * x_c       # self*
            # x_c_self = torch.cat(x_c, torch.flatten(x_c_self))
            if t >= ig:
                x_loss += torch.sum(torch.pow(x - x_h, 2) * m) / (torch.sum(m) + 1e-5)  # loss of known values

            inputs = torch.cat([x_c, m], dim=1)  # cat

            c = self.c_decay(c, D)

            h, c = self.rnn_cell(inputs, (h, c))
            h_n[:, t, :] = h

            imputations.append(x_c.unsqueeze(dim=1))

        mean_x_loss = x_loss / pred_len
        imputations = torch.cat(imputations, dim=1)

        # y_h = self.out(h)
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)        # predicthon loss

        # only use training labels
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        # y_h = F.sigmoid(y_h)        # classification

        return {'loss': mean_x_loss * self.impute_weight,  # + y_loss * self.label_weight
                # 'predictions': y_h,
                'imputations': imputations,
                # 'labels': labels,
                # 'is_train': is_train,
                # 'evals': evals,
                # 'eval_masks': eval_masks,
                'h_n': h_n}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
