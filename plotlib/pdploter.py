import copy
import numpy as np
import random
import matplotlib.pyplot as plt


"""
00 = {str_: ()} Date
01 = {str_: ()} Coke\n(/64)\n(inc)
02 = {str_: ()} Fre.\n(z)
03 = {str_: ()} Working\nours\n()
04 = {str_: ()} Production (measured)
05 = {str_: ()} Welle\nPressure\n(psi)
06 = {str_: ()} Casing\nPressure\n(psi)
07 = {str_: ()} Back\nPressure\n(psi)
08 = {str_: ()} stopTime
09 = {str_: ()} subMeasuring
10 = {str_: ()} openWell
11 = {str_: ()} Date0
12 = {str_: ()} Coke\n(/64)\n(inc)0
13 = {str_: ()} Fre.\n(z)0
14 = {str_: ()} Working\nours\n()0
15 = {str_: ()} Production (measured)0
16 = {str_: ()} Production (Reduced)0
17 = {str_: ()} Welle\nPressure\n(psi)0
18 = {str_: ()} Casing\nPressure\n(psi)0
19 = {str_: ()} Back\nPressure\n(psi)0
20 = {str_: ()} stopTime0
21 = {str_: ()} subMeasuring0
22 = {str_: ()} openWell0
23 = {str_: ()} cNotChosen0
"""


def plot_xy(x, y, target, opt, well_name):
    # 绘制数据时序分布
    fx0 = []
    fx1 = []
    fy0 = []
    fy1 = []
    up_opt = []
    down_opt = []
    for e_date, e_measure in zip(x, y):
        fx0.append(int(e_date))
        fy0.append(float(e_measure))

    for e_date, e_opt in zip(x, opt):
        if e_opt[1] > 0 or e_opt[2] > 0:
            up_opt.append(e_date)
        if e_opt[1] < 0 or e_opt[2] < 0:
            down_opt.append(e_date)

    fx0 = np.array(fx0)
    fx1 = np.array(fx1)
    fy0 = np.array(fy0)
    fy1 = np.array(fy1)

    for eachup_opt, eachdown_opt in zip(up_opt, down_opt):
        plt.plot([eachup_opt, eachup_opt], [-4000, 4000], c='red')
        plt.plot([eachdown_opt, eachdown_opt], [-4000, 4000], c='green')
        # plt.show()

    plt.plot(fx0, fy0, marker='.', c='blue')
    # plt.scatter(fx1, fy1, marker='x', c='red')
    plt.xlabel('date')
    plt.ylabel('measured')
    plt.title(well_name)

    plt.axis([-100, 2000, -100, 3500])
    plt.show()


def plotMeasure_date(date, measure, target, reduce, opt, well_name):
    # 绘制数据时序分布
    fx0 = []
    fx1 = []
    fy0 = []
    fy1 = []
    up_opt = []
    down_opt = []
    for e_date, e_measure, e_target, e_reduce, e_opt in zip(date, measure, target, reduce, opt):
        if e_opt[1] > 0 or e_opt[2] > 0:
            up_opt.append(e_date)
        if e_opt[1] < 0 or e_opt[2] < 0:
            down_opt.append(e_date)
        fx0.append(e_date)
        fy0.append(e_measure)
        if e_target[0] == 1:
            fx1.append(e_date)
            fy1.append(e_measure)

    fx0 = np.array(fx0)
    fx1 = np.array(fx1)
    fy0 = np.array(fy0)
    fy1 = np.array(fy1)

    for eachup_opt, eachdown_opt in zip(up_opt, down_opt):
        plt.plot([eachup_opt, eachup_opt], [-4000, 4000], c='red')
        plt.plot([eachdown_opt, eachdown_opt], [-4000, 4000], c='green')
        # plt.show()

    plt.plot(fx0, fy0, marker='.', c='blue')
    plt.scatter(fx1, fy1, marker='x', c='red')
    plt.xlabel('date')
    plt.ylabel('measured')
    plt.title(well_name)

    plt.axis([-100, 2000, -100, 3500])
    plt.show()


def plotMeasureDifference_date(date, difference, target, reduce, opt, well_name):
    # 绘制数据时序分布
    fx0 = []
    fx1 = []
    fy0 = []
    fy1 = []
    up_opt = []
    down_opt = []
    for e_date, e_measure, e_target, e_reduce, e_opt in zip(date, difference, target, reduce, opt):
        if e_opt[1] > 0 or e_opt[2] > 0:
            up_opt.append(e_date)
        if e_opt[1] < 0 or e_opt[2] < 0:
            down_opt.append(e_date)
        fx0.append(e_date)
        fy0.append(e_measure)
        if e_target[0] == 1:
            fx1.append(e_date)
            fy1.append(e_measure)

    fx0 = np.array(fx0)
    fx1 = np.array(fx1)
    fy0 = np.array(fy0)
    fy1 = np.array(fy1)

    plt.plot([0, 4000], [0, 0], c='black')

    for eachup_opt, eachdown_opt in zip(up_opt, down_opt):
        plt.plot([eachup_opt, eachup_opt], [-3000, 3000], c='red')
        plt.plot([eachdown_opt, eachdown_opt], [-3000, 3000], c='green')
        # plt.show()

    plt.plot(fx0, fy0, marker='.', c='blue')
    plt.scatter(fx1, fy1, marker='x', c='red')
    plt.xlabel('date')
    plt.ylabel('difference')
    plt.title(well_name)

    plt.axis([-100, 2000, -2000, 2000])
    plt.show()


def plotMeasure_reduce(Data, target, pre, now, temp):
    # 绘制生成数据二维分布
    fx0 = []
    fx1 = []
    fy0 = []
    fy1 = []
    x1 = []
    y1 = []
    x0 = []
    y0 = []
    t = np.array(temp)
    for tempy, templ in zip(Data, target):
        if templ[0] == 1:
            fx1.append(tempy[pre])
            fy1.append(tempy[now])
        else:
            fx0.append(tempy[pre])
            fy0.append(tempy[now])
    for each in temp:
        if each[2] == 1:
            x1.append(each[0])
            y1.append(each[1])
        else:
            x0.append(each[0])
            y0.append(each[1])
    fx0 = np.array(fx0)
    fx1 = np.array(fx1)
    fy0 = np.array(fy0)
    fy1 = np.array(fy1)
    # plt.scatter(fx0, fy0, marker='.', c='green')
    # plt.scatter(fx1, fy1, marker='.', c='red')
    plt.scatter(x1, y1, marker='x', c='red')
    plt.scatter(x0, y0, marker='x', c='blue')
    plt.xlabel('reduced')
    plt.ylabel('measured')
    # plt.plot(selectedIndex, y, marker='.', c='green')
    # plt.scatter(selectedIndex, tempData.data[:, idxWaterCutInUseWash], '-', c='red')
    plt.axis([-100, 3100, -100, 3100])
    plt.show()


class WellDataPloter:
    matchScoreArray = []
    mData = []
    mFeatureNames = []

    formedData = []
    formedTarget = []
    formedIndex = []

    def __init__(self, data=0, featureNames=0, target=0, wellname='well-xx'):
        self.mdata = data
        self.mFeatureNames = list(featureNames)
        self.mtarget = target
        self.well_name = wellname


    def getNameIndex(self, findName):
        idx = self.mFeatureNames.index(findName)
        return idx


    def plot_measure_opt_interpolate(self):
        sum = 0
        nowFreIdx = self.getNameIndex('Fre.\n(z)')
        preFreIdx = self.getNameIndex('Fre.\n(z)01')
        nowChokeIdx = self.getNameIndex('Coke\n(/64)\n(inc)')
        preChokeIdx = self.getNameIndex('Coke\n(/64)\n(inc)01')
        nowFlIdx = self.getNameIndex('Production (measured)')
        nowFlIdx_r = self.getNameIndex('Production (Reduced)')

        opt = []
        difference = []
        idx = 0
        for eachData in self.mdata:
            idx += 1

        print('fre num: ' + str(sum))
        # 绘制分布图像
        plotFormedData = 1
        if plotFormedData:
            plot_xy(self.mdata[:, 0], self.mdata[:, nowFlIdx], self.mtarget, opt, self.well_name)
        return

    def plot_measure_opt(self):
        sum = 0
        nowFreIdx = self.getNameIndex('Fre.\n(z)')
        preFreIdx = self.getNameIndex('Fre.\n(z)01')
        nowChokeIdx = self.getNameIndex('Coke\n(/64)\n(inc)')
        preChokeIdx = self.getNameIndex('Coke\n(/64)\n(inc)01')
        nowFlIdx = self.getNameIndex('Production (measured)')
        preFlIdx_m= self.getNameIndex('Production (measured)0')
        preFlIdx_r = self.getNameIndex('Production (Reduced)0')

        opt = []
        difference = []
        idx = 0
        for eachData in self.mdata:
            nowChoke = eachData[nowChokeIdx]
            preChoke = eachData[preChokeIdx]
            subChoke = nowChoke - preChoke
            absSubChoke = abs(subChoke)

            nowFre = eachData[nowFreIdx]
            preFre = eachData[preFreIdx]
            subFre = nowFre - preFre
            absSubFre = abs(subFre)

            difference.append(eachData[nowFlIdx] - eachData[preFlIdx_m])
            opt.append([eachData[0], subFre, subChoke])
            idx += 1

        print('fre num: ' + str(sum))
        # 绘制分布图像
        plotFormedData = 1
        if plotFormedData:
            # plotMeasureDifference_date(self.mdata[:, 0], difference, self.mtarget, self.mdata[:, preFlIdx_r], opt, self.well_name)
            plotMeasure_date(self.mdata[:, 0], self.mdata[:, nowFlIdx], self.mtarget, self.mdata[:, preFlIdx_r], opt, self.well_name)
        return
