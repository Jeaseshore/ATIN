import numpy as np
from src.pack_toolkit.pack import Bunch

def getNameIndex(names, findName):
    name_list = list(names)
    idx = name_list.index(findName)
    return idx

def delete_features(dataset, delete_columns):
    dataset.data = np.delete(dataset.data, delete_columns, axis=1)
    dataset.feature_names = np.delete(dataset.feature_names, delete_columns)
    return dataset.data, dataset.feature_names

