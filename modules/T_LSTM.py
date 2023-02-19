import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from src.loss.lossFunction import py_sigmoid_focal_loss
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
        c_s = c_s / torch.log(torch.e + D)
        c_hat = c_l + c_s

        return c_hat


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
        self.c_decay = CellStateT(self.rnn_hid_size)
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
        self.lossfunc = py_sigmoid_focal_loss

    def forward(self, data, direct='forward'):
        # Original sequence
        data['forward']['sLabels'][:, -1] = 0
        data['backward']['sLabels'][:, -1] = 0
        values = data[direct]['values']
        masks = data[direct]['masks']
        Deltas = data[direct]['Deltas']  # [batch_size, seq_len, 1]
        # ys = data[direct]['sLabels']

        label = data['labels'].unsqueeze(-1)  # [batch_size,]
        label_ex = 1 - label
        label_ex = torch.concat((label, label_ex), dim=-1)

        SEQ_LEN = values.size()[1]

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))

        if torch.cuda.is_available():
            h, c, h_n = h.cuda(), c.cuda(), h_n.cuda()

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            D = Deltas[:, t]
            # y = ys[:, t, :]

            x_c = m * x + (1 - m) * 0  # fill missing
            inputs = x_c
            # inputs = torch.cat([x_c, y], dim=1)  # cat y
            c = self.c_decay(c, D)
            h, c = self.rnn_cell(inputs, (h, c))
            h_n[:, t, :] = h

        pred = self.re(h)
        loss = self.lossfunc(pred, label_ex)

        return {'lossPred': loss,
                'lossImputation': torch.tensor(.0),
                'h_n': h_n,
                'pred': pred,
                'labels': label_ex}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['lossPred'].backward()
            optimizer.step()

        return ret


if __name__ == '__main__':
    batch_seq = torch.randn(16, 8, 6).cuda()
    deltas = torch.ones(16, 8, 1).cuda()
    mask = torch.ones(16, 8, 6).cuda()
    labels = torch.zeros(16, 8, 2).cuda()
    data = {'forward': {'values': batch_seq, 'Deltas': deltas, 'masks': mask, 'sLabels': labels}}
    m = Model(6, 32).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    h_n = m(data, 'forward')
    loss = torch.sum(h_n[:, -1, :] - h_n[:, -1, :] / 2)
    loss.backward()
    opt.step()

