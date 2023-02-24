import torch
import torch.nn as nn

from torch.autograd import Variable

# import src.modules.attain_seq_global as attain
import src.modules.attem_b as attem_b
import src.modules.attem_f as attem_f
from utils import reverse_tensor
from  utils import reverse

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
        # self.attain = attain.Model(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.attem_bw = attem_b.Model(self.input_size, self.rnn_hid_size)
        self.attem_fw = attem_f.Model(self.input_size, self.rnn_hid_size)

    def forward(self, data):
        # a_tj_reverse = self.reverse_tensor(a_tj)
        # ret_b = self.attain(data, 'forward')

        # ys = data['forward']['sLabels']
        # data['forward']['values'] = torch.cat([data['forward']['values'], ys], dim=-1)
        data['backward']['values'] = reverse_tensor(data['forward']['values'])

        ret_b = self.attem_bw(data, 'backward')
        # data['backward'] = reverse(data['backward'])
        ret_f = self.attem_fw(data, 'forward')

        return {'h_n_forward': ret_f, 'h_n_backward': ret_b}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
