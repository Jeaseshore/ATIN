import warnings

import numpy as np
from src.file.readfile import loadCSVDataset
import math


# diff[i] = data[i] - data[i - 1]
def difference(input, order=1):
    """
    求差分
    :param input: 输入数据， ndarray float
    :param dim: 差分维度，默认为第 1 维度
    :param order: 差分阶数， 默认为 1。 阶数数量的第一个起始数据无法求差分
    :return: diff 差分结果
    """
    num_sample = len(input)
    # num_features = len(input[0, :])
    diff = np.array(input.astype(float))
    for j in range(order):
        for i in range(num_sample - 1, 0, -1):
            diff[i] = diff[i] - diff[i - 1]

    return diff


def fill_miss(input, method='avg'):
    """
    处理缺省值
    :param input:
    :return:
    """
    data = np.array(input.astype(float))
    num_sample = len(input[:, 0])
    num_features = len(input[0, :])
    if method == 'avg':
        fills = feature_avg(data)
    elif method == 'mid':
        fills = feature_mid(data)
    else:  # ?
        fills = interpolation(data)

    for i in range(num_sample):
        for j in range(num_features):
            if math.isnan(data[i, j]):
                data[i, j] = fills[j]
    return data


import statistics
def feature_avg(input):
    """
    求各个特征均值
    :param input: ndarray, float
    :return:
    """
    data = np.array(input.astype(float))
    num_sample = len(input[:, 0])
    num_features = len(input[0, :])
    f_mean = np.zeros(num_features)
    f_num = np.zeros(num_features)
    for i in range(num_sample):
        for j in range(num_features):
            value = data[i, j]
            if math.isnan(value):
                continue
            else:
                f_mean[j] += value
                f_num[j] += 1
    f_mean = f_mean / f_num
    return f_mean


def feature_mid(input):
    """
    求各个特征中值
    :param input: ndarray, float
    :return:
    """
    data = np.array(input.astype(float))
    num_sample = len(input[:, 0])
    num_features = len(input[0, :])
    mid = np.zeros(num_features)
    for i in range(num_features):
        mid[i] = statistics.median(data[:, i])
    for i in range(num_features):
        if math.isnan(mid[i]):
            warnings.warn('出现 nan 值')
    return mid


def interpolation(input, method='Lagrange'):
    pass


def diff_test(data):
    """
        差分测试
    """
    diff = difference(data)  # 去除最后三行空值， 前两列时间和监测点数据
    pass


def avg_test(data):
    """
        均值测试
    """
    res = feature_avg(data)
    pass


def mid_test(data):
    """
        中值测试
    """
    res = feature_mid(data)
    pass


if __name__ == '__main__':
    # step.1 读取数据
    filename = '../data/Appendix1_PointA_perDay_measure.csv'
    dataset = loadCSVDataset(filename)
    feature_names, data = dataset.feature_names, dataset.data
    data = data[0:-3, 2:]  # 去除最后三行空值， 前两列时间和监测点数据
    # step.2 测试
    # diff_test(data)
    avg_test(data)
    # mid_test(data)
    pass
