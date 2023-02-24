import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from src.preprocessing.normalizer import NdarrayNormalizer

"""
    seqdata
    seqmask
    seqDeltaF
    seqDelatB
    Deltas_f
    Deltas_b
    label
    date
"""
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.content = data

        keys = list(self.content.keys())
        for i in range(len(keys) - 1):
            templen1 = len(self.content[keys[i]])
            templen2 = len(self.content[keys[i + 1]])
            if templen2 != templen1:
                print("error, data length is not same")

    def __len__(self):
        return len(self.content[list(self.content.keys())[0]])

    def __getitem__(self, idx):
        ret = {}
        for key in self.content:
            ret.update({key: self.content[key][idx]})
        return ret


"""
    'seqdata': seqData, 'seqmask': seqMask,
    'seqdelta_f': seqdelta_f, 'seqdelat_b': seqdelat_b,
    'seqDeltas_F': seqDeltas_F, 'seqDeltas_B': seqDeltas_B,
    'seqLabel': seqLabel, 'date': seqDate
    'labels'
"""
def collate_fn(recs):
    values = torch.FloatTensor(np.array(list(map(lambda r: r['seqdata'], recs))))
    masks = torch.FloatTensor(np.array(list(map(lambda r: r['seqmask'], recs))))
    deltas = torch.FloatTensor(np.array(list(map(lambda r: r['seqdelta_f'], recs))))
    forwards = torch.FloatTensor(np.array(list(map(lambda r: r['seqforwards_f'], recs))))
    Deltas = torch.FloatTensor(np.array(list(map(lambda r: r['seqDeltas_F'], recs))))
    seqLabel = torch.FloatTensor(np.array(list(map(lambda r: r['seqLabel'], recs))))

    forward = {'values': values,
               'masks': masks,
               'deltas': deltas,
               'forwards': forwards,
               'Deltas': Deltas,
               'sLabels': seqLabel
               }

    forwards = torch.FloatTensor(np.array(list(map(lambda r: r['seqforwards_b'], recs))))
    deltas = torch.FloatTensor(np.array(list(map(lambda r: r['seqdelat_b'], recs))))
    Deltas = torch.FloatTensor(np.array(list(map(lambda r: r['seqDeltas_B'], recs))))
    backward = {'values': values,
                'masks': masks,
                'deltas': deltas,
                'forwards': forwards,
                'Deltas': Deltas,
                'sLabels': seqLabel
                }

    ret_dict = {'forward': forward, 'backward': backward}

    ret_dict['labels'] = torch.FloatTensor(np.array(list(map(lambda x: x['labels'], recs))))    # fluid only
    return ret_dict


def data_reduce(data, is_reduce=True):
    """
       ['Coke\n(/64)\n(inc)' 'Fre.\n(z)' 'Fluid Rate0\n(bbl/d)', 'Fluid Rate\n(bbl/d)'
       'Measured\n(%)' 'Dece\n(%)', 'Welle\nPressure\n(psi)' 'Tem.\n(℃)']
    """
    if is_reduce:
        delete_column = [2, 4]
    else:
        delete_column = [3, 5]
    keys = ['seqdata', 'seqmask', 'seqdelta_f', 'seqdelat_b', 'seqforwards_f', 'seqforwards_b']
    for key in keys:
        data[key] = np.delete(data[key], delete_column, axis=-1)
    return data


def data_fluid(data, is_fluid=True):
    if is_fluid:
        lb = 0
    else:
        lb = 1
    label = data['seqLabel'][:, -1, lb]
    data['labels'] = label
    index = []
    for idx, eachlabel in enumerate(label):
        if eachlabel >= 0:
            index.append(idx)
    for key in data:
        data[key] = data[key][index]
    return data


# 获取标准化后的数据
def get_data_dict(data_dict, is_fluid=True, is_reduce=False):
    del data_dict['featureNames']
    norm = NdarrayNormalizer(method='min-max')
    data_dict['seqdata'] = norm.normalization(data_dict['seqdata'])  # normalization with nan
    data_dict = data_fluid(data_dict, is_fluid)
    data_dict = data_reduce(data_dict, is_reduce)
    sortDataset(data_dict)
    return data_dict


def split_set(data_dict, rate=0.2):
    length = len(data_dict['date'])
    idx = int(length * rate)
    train_set = {}
    test_set = {}
    for key in data_dict:
        test_set.update({key: data_dict[key][:idx]})
        train_set.update({key: data_dict[key][idx:]})
    return train_set, test_set


def sortDataset(data_dict):
    sortkey = 'date'
    indices = np.argsort(data_dict[sortkey][:, -1])
    for key in data_dict:
        data_dict[key] = data_dict[key][indices]
    return data_dict


def get_loader(data_dict, batch_size=1, shuffle=True):
    data_set = MyDataset(data_dict)
    data_iter = DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return data_iter


if __name__ == '__main__':
    data = np.load('./tempdata/wells_data_mask_onlymeasure.npy', allow_pickle=True).item()
    del data['featureNames']

    # sortDataset(data)
    # del data['date']

    split_set(data)
    loader = get_loader(data, batch_size=32)
    for idx, eachdata in enumerate(loader):
        # print(eachdata)
        pass
    pass
