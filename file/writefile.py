import csv
import numpy as np


def writecsvdata(data, filename):
    """
    写数据
    :param data:
    :return:
    """
    file = open(filename, 'w', encoding='UTF-8', newline='')
    writer = csv.writer(file)
    for each_row in data:
        writer.writerow(each_row)
    file.close()
