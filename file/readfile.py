import csv
import numpy as np
from src.pack_toolkit.pack import Bunch
import os

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    return files

def load_filenames(dir):
    files = file_name(dir)
    filenames = []
    file_paths = []
    for eachFile in files:
        filenames.append(eachFile)
        eachFile = dir + eachFile
        file_paths.append(eachFile)
    return file_paths, filenames


def loadCSVDataset(loaderfilename):
    csv_reader = csv.reader(open(loaderfilename, 'r'))
    data = []
    dataOrigin = []
    idxFirstCol = 0
    idxLastCol = 28  # excel 批量生成csv文件存在问题， 读取csv文件时指定读取到第28列

    firstRow = next(csv_reader)
    # next(csv_reader)
    feature_names = firstRow[idxFirstCol:idxLastCol]

    for row in csv_reader:
        dataOrigin.append(row[idxFirstCol:idxLastCol])
        if row[0] == '':
            break
        for i in range(1, idxLastCol - 1):
            if row[i] == '' or row[i] == ' ':
                row[i] = 'nan'
        data.append(row[idxFirstCol:idxLastCol])

    data = np.array(data)
    feature_names = np.array(feature_names)

    dataSet = Bunch(
        data=data,
        feature_names=feature_names
    )
    return dataSet


if __name__ == '__main__':
    data = loadCSVDataset('../../../data/wells/wellsAD1-1-4H.csv')
    filenames = load_filenames('../../../data/wells')
    pass
