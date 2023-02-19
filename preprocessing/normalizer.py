import numpy as np
import torch


class TensorNormalizer:
    """
       torch.tensor 类型归一化
    """
    def __init__(self):
        self.tmin = None
        self.tmax = None
        self.scaler = None

    def normalization(self, dataMatrix, nordim=0):
        """
        对tensor类型归一化
        :param dataMatrix: tensor 数据矩阵
        :param nordim: 归一化维度
        :return: tensor 归一化结果
        """
        if self.scaler == None:
            self.tmin = torch.amin(dataMatrix, nordim)
            self.tmax = torch.amax(dataMatrix, nordim)
            self.scaler = torch.sub(self.tmax, self.tmin)
        offset = torch.sub(dataMatrix, self.tmin)
        res = torch.div(offset, self.scaler)
        res = torch.nan_to_num(res)
        return res


def normalizationTensor(dataSet):
    """
    tensor 类型单次归一化
    :param dataSet:  数据矩阵
    :return: 归一化结果
    """
    tmin = torch.amin(dataSet, 0)
    tmax = torch.amax(dataSet, 0)
    scaler = torch.sub(tmax, tmin)
    offset = torch.sub(dataSet, tmin)
    res = torch.div(offset, scaler)
    res = torch.nan_to_num(res)
    return res


class NdarrayNormalizer:
    """
    ndarray 归一化
    """
    def __init__(self, method='min-max'):
        self.min = None
        self.max = None
        self.scaler = None
        self.std_dev = None
        self.mean = None
        self.method = method
        self.is_norm = False

    def normalization(self, dataMatrix, nordim=0):
        """
        归一化， 含 nan
        :param dataMatrix: ndarray 数据矩阵
        :param nordim: 归一化维度
        :param method: 归一化方式 'min-max', 'z-score'
        :return: ndarray
        """
        if self.method == 'min-max':
            if self.is_norm == False:
                self.min = np.nanmin(dataMatrix, nordim)
                self.max = np.nanmax(dataMatrix, nordim)
                self.scaler = self.max - self.min
                self.is_norm = True
            res = (dataMatrix - self.min) / self.scaler
        elif self.method == 'z-score':
            if self.is_norm == False:
                self.mean = np.nanmean(dataMatrix, nordim)
                self.std_dev = np.nanstd(dataMatrix, nordim)
                self.is_norm = True
            res = (dataMatrix - self.mean) / self.std_dev
        return res

    def denormal(self, dataMatrix, nordim=0):
        if self.method == 'min-max':
            res = dataMatrix * self.scaler + self.min

        elif self.method == 'z-score':
            res = dataMatrix * self.std_dev + self.mean
        return res


def normalizing(x, method='min-max', dim=0):
    """
    ndarray 单次归一化
    :param x: ndarray
    :param dim: 归一化维度
    :return: 归一化结果
    """
    if method == 'min-max':
        max = np.nanmax(x, dim)
        min = np.nanmin(x, dim)
        x = (x - min) / (max - min)
    elif method == 'z-score':
        sigma = np.nanstd(x, dim)
        mu = np.nanmean(x, dim)
        x = (x - mu) / sigma
    return x


from src.file.readfile import loadCSVDataset
from src.preprocessing.handling_miss import fill_miss

def tensornormalization_test():
    filename = '../data/Appendix1_PointA_perDay_measure.csv'
    dataset = loadCSVDataset(filename)
    f, data = dataset.feature_names, dataset.data
    data = data[0:-3, 2:]
    data = fill_miss(data)
    data = torch.tensor(data)

    ndnorm = TensorNormalizer()
    data_norm = ndnorm.normalization(data)
    pass


def ndnormalization_test():
    filename = '../data/Appendix1_PointA_perDay_measure.csv'
    dataset = loadCSVDataset(filename)
    f, data = dataset.feature_names, dataset.data
    data = data[0:-3, 2:]

    ndnorm = NdarrayNormalizer()
    data_norm = ndnorm.normalization(data, method='z-score')
    pass


if __name__ == '__main__':
    ndnormalization_test()
    # tensornormalization_test()
    pass
