import torch
import torch.nn as nn

from torch.autograd import Variable

import src.modules.attain_seq_global as attain
from utils import reverse_tensor
from utils import reverse
from src.loss.lossFunction import py_sigmoid_focal_loss


SEQ_LEN = 36


class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight=1, label_weight=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        # self.rits_f = ritsi_dec.Model(self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.attain = attain.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        # self.attem_bw = attem_b.Model(self.input_size, self.rnn_hid_size)
        # self.attem_fw = attem_f.Model(self.input_size, self.rnn_hid_size)
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

    def forward(self, data):
        # a_tj_reverse = self.reverse_tensor(a_tj)
        """
        data['forward']['sLabels'][:, -1] = 0
        data['backward']['sLabels'][:, -1] = 0
        ys = data['forward']['sLabels']
        data['forward']['values'] = torch.cat([data['forward']['values'], ys], dim=2)
        # data['backward']['values'] = reverse_tensor(data['forward']['values'])
        """

        label = data['labels'].unsqueeze(-1)  # [batch_size,]
        label_ex = 1 - label
        label_ex = torch.concat((label, label_ex), dim=-1)

        h_n = self.attain(data, 'forward')
        pred = self.re(h_n[:, -1, :])

        loss = self.lossfunc(pred, label_ex)

        return {'lossImputation': torch.tensor([0]), 'lossPred': loss, 'pred': pred, 'labels': label_ex}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['lossPred'].backward()
            optimizer.step()

        return ret
