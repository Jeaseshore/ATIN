"""
ATTAIN seq global network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

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


class Attain_alpha(nn.Module):
    def __init__(self):
        super(Attain_alpha, self).__init__()


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
        self.softmax = torch.nn.Softmax(dim=-1)
        self.atten_linear = torch.nn.Linear(self.input_size, self.input_size)

    def forward(self, data, direct):
        values = data[direct]['values']
        Deltas = data[direct]['Deltas']     # [batch_size, seq_length, 1]
        ys = data[direct]['sLabels']
        SEQ_LEN = values.size()[1]

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        # h2 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        # c2 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))
        c_n = torch.zeros((values.size()[0], SEQ_LEN, self.rnn_hid_size))

        if torch.cuda.is_available():
            h, c, h_n, c_n = h.cuda(), c.cuda(), h_n.cuda(), c_n.cuda()
            # h2, c2 = h2.cuda(), c2.cuda()

        Deltas = torch.cumsum(Deltas, dim=-2)                           # cumsum seq [batch_size, seq_length, 1]
        Deltas = Deltas.repeat_interleave(len(Deltas[0] - 1), dim=-1)   # repeat seq [batch, seq_length, seq_length]
        Deltas_T = torch.transpose(Deltas, -1, -2)
        D = Deltas - Deltas_T                                           # D [batch, seq_length, seq_length]

        # a_ij = x_i^T * W_a * x_j
        data_T = torch.transpose(values, -1, -2)  # [batch_size, feature_size, seq_length]
        values = self.atten_linear(values)
        a_tj = torch.bmm(values, data_T)          # [batch_size, seq_length, seq_length]

        inputs = values[:, 0, :]
        # inputs = torch.cat([inputs, ys[:, 0, :]], dim=1)
        h, c = self.rnn_cell(inputs, (h, c))
        h_n[:, 0, :] = h
        c_n[:, 0, :] = c
        for t in range(1, SEQ_LEN, 1):
            x = values[:, t, :]       # [batch_size, input_size]
            a = a_tj[:, t, :t]      # [batch_size, t]
            d = D[:, t, :t]         # [batch_size, t]
            y = ys[:, t, :]
            # g(∆t) = 1 / log(e + ∆t)
            # alpha = g * a
            # d = 1 / torch.log(1 + d)  # d > 0
            # alpha = self.softmax(a * d)     # [batch_size, t]
            alpha = self.softmax(a)  # [batch_size, t]
            alpha = alpha * 1 / torch.log(torch.e + d)  # d > 0
            c_mul_alpha = c_n[:, :t, :].clone() * alpha.unsqueeze(-1)   # [batch_size, t, hidden] * [batch_size, t, 1] = [batch_size, t, hidden]
                                                                        # c_n[:t].clone() is needed in loop
            C = torch.sum(c_mul_alpha, dim=-2)                          # [batch_size, t, hidden] -> [batch_size, hidden]
            # inputs = torch.cat([x, y], dim=1)
            h, c = self.rnn_cell(x, (h, C))
            h_n[:, t, :] = h
            c_n[:, t, :] = c

        return h_n

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


if __name__ == '__main__':
    m = Model(input_size=6, rnn_hid_size=32).cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(4, 16, 6).cuda()
    x.requires_grad = True
    D = torch.randn(4, 16, 1).cuda()
    D = torch.abs(D)            # Deltas >= 0

    h_n = m(x, D)
    loss = torch.sum(h_n[:, -1, :] - h_n[:, -1, :]/2)
    loss.backward()
    optimizer.step()
    pass