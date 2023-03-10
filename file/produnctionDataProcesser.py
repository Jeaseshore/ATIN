import copy
import numpy as np
# import pandas as pd

from abandoned_code.formKrlData import formRLData

class WellGroups:
    wellID = []
    dataProcessers = []
    pFeatures = []
    pDataAll = []
    pTargetAll = []

    def __init__(self, fileNames, dformat=0, numDataItems=4):
        for eachFile in fileNames:
            tempPDP = ProductionDataProcesser(eachFile, dformat, numDataItems)
            WellGroups.dataProcessers.append(tempPDP)

    def loadAllData(self):
        flag = 0
        for eachPDP in WellGroups.dataProcessers:
            dataset = eachPDP.getDataset()[1]
            if len(dataset.data) == 0:
                print('remove Well: ' + eachPDP.well_name)
                continue

            singleWellData, singleWellTarget = dataset.data, dataset.target
            if flag == 0:
                WellGroups.pFeatures = dataset.feature_names
                WellGroups.pDataAll = singleWellData
                WellGroups.pTargetAll = singleWellTarget
                flag = 1
            else:
                WellGroups.pDataAll = np.append(WellGroups.pDataAll, singleWellData, axis=0)
                WellGroups.pTargetAll = np.append(WellGroups.pTargetAll, singleWellTarget, axis=0)
            WellGroups.wellID.append(singleWellData[0, 1])


    def loadData_wells(self):
        flag = 0
        for eachPDP in WellGroups.dataProcessers:
            dataset = eachPDP.getDataset()[1]
            if len(dataset.data) == 0:
                print('remove Well: ' + eachPDP.well_name)
                continue

            singleWellData, singleWellTarget = dataset.data, dataset.target
            if flag == 0:
                WellGroups.pFeatures = dataset.feature_names
                WellGroups.pDataAll = singleWellData
                WellGroups.pTargetAll = singleWellTarget
                flag = 1
            else:
                WellGroups.pDataAll = np.append(WellGroups.pDataAll, singleWellData, axis=1)
                WellGroups.pTargetAll = np.append(WellGroups.pTargetAll, singleWellTarget, axis=1)
            WellGroups.wellID.append(singleWellData[0, 1])

    def formatNewData(self, data):
        newDataWellId = data[1]
        indexPDP = WellGroups.wellID.index(newDataWellId)
        lastWellData = WellGroups.dataProcessers[indexPDP]
        return

    def addFile(self, addFileName):
        return

    def addData(self, wellID, data):
        return

    def getDataAll(self):
        self.loadAllData()
        return WellGroups.pDataAll, WellGroups.pTargetAll

    def get_ts_wells(self):
        self.loadData_wells()
        return WellGroups.pDataAll, WellGroups.pTargetAll


def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False

remarksKey = ['Measuring', 'Sampling', 'well shut in', 'open well', 'stop pump']

class ProductionDataProcesser:
    def __init__(self, filename='../data/productionData.csv', dformat=0, numDataItems=4):
        temp_name = filename.split("/")[-1]
        temp_name = temp_name.replace('.csv', '')
        self.well_name = temp_name

        self.dformat = dformat
        self.numDataItems = numDataItems
        self.loaderfilename = filename
        self.measureStr = remarksKey[0]  # ?????????????????????
        self.sampleStr = remarksKey[1]  # ?????????????????????
        self.openWellStr = remarksKey[3]  # ???????????????
        # ????????????
        idxFluidRatInUse = 12  # ???????????????????????? ??????
        idxWaterCutInUse = 16  # ???????????????????????? ??????
        idxMeasureF = 9  # ??????????????? ??????
        idxMeasureW = 15  # ??????????????? ??????

        # ????????????
        self.delete_columns = [1, 2, 3, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

        # ?????????????????????
        offsetWash = 3 + 2
        self.idxFluidRatInUseWash = idxFluidRatInUse - offsetWash - 2  # ???????????????????????? ??????
        self.idxWaterCutInUseWash = idxWaterCutInUse - offsetWash - 4  # ???????????????????????? ??????
        self.idxFluidRatMeasureWash = idxMeasureF - offsetWash  # ??????????????? ??????
        self.idxWaterCutSampleWash = idxMeasureW - offsetWash - 4  # ??????????????? ??????
        self.idxWorkingHours = 8 - offsetWash

    def getDataset(self):
        self.loadCSVDataset()
        self.dataFormatHalfPre()
        # self.addFormedData()
        self.dataFormatHalfNext()
        return self.selectedIndex, self.tempData

    def get_TS_interpolate(self):
        self.loadCSVDataset()
        self.dataFormatHalfPre()
        # self.addFormedData()
        self.dataFormatHalfNext()
        return self.selectedIndex, self.tempData

    def handlingNan(self):
      return

    def get_sample_points(self):
        # Step 0.
        # ??????
        measureStr = self.measureStr
        sampleStr = self.sampleStr
        openWellStr = self.openWellStr

        tempData = copy.deepcopy(self.dataSet)
        # ????????????
        delete_columns = self.delete_columns

        idxFluidRatInUseWash = self.idxFluidRatInUseWash
        idxWaterCutInUseWash = self.idxWaterCutInUseWash
        idxFluidRatMeasureWash = self.idxFluidRatMeasureWash
        idxWaterCutSampleWash = self.idxWaterCutSampleWash
        idxWorkingHours = self.idxWorkingHours

        selectedData = []  # list ???????????????????????????
        labelFormatted = self.targetFormatted = []  # list ?????????????????????(target)
        selectedIndex = self.selectedIndex = []  # ??????2???????????????????????????

        tempData.data = np.delete(tempData.data, delete_columns, axis=1)
        tempData.feature_names = np.delete(tempData.feature_names, delete_columns)

        idx = 0
        preM = 0
        preS = 0
        addInfo = []
        cNotChosen = [0, 0]
        now = 0
        pre = 1
        # Step 1.
        # ???????????????????????????????????????
        for row in tempData.data:
            if idx < self.numDataItems:
                idx += 1
                continue
            matchMeasure = re.match(measureStr, row[-1], re.M | re.I)  # ???????????????
            matchSample = None  # re.match(sampleStr, row[-1], re.M | re.I)  # ???????????????
            matchOpenWell = re.search(openWellStr, row[-1], re.M | re.I)  # ???????????????

            if matchOpenWell != None:
                openWell = 1
            else:
                openWell = 0
            # ?????????????????????row ?????????
            if matchSample == None and matchMeasure == None:
                idx += 1
                continue

            # ????????????????????????
            subM = 0
            subS = 0

            # remark?????? ??????????????????
            stopTime = getStopTime(row[-1])
            Date = getDateInterval(row[0])
            row[0] = Date
            workingHours = float(row[idxWorkingHours])
            realWT = workingHours - stopTime
            dayHours = 24

            targetF = 0
            targetW = 0

            # ???????????????
            if matchMeasure:
                subM = idx - preM
                preM = idx

                # ?????????(????????????)
                nowMeasureF = float(row[idxFluidRatMeasureWash])  # ???????????????
                nowInUseF = float(row[idxFluidRatInUseWash])
                changeRate = abs(float(nowInUseF) / realWT - float(nowMeasureF) / dayHours)  # ??????

                # row[idxFluidRatMeasureWash] = float(nowMeasureF)/dayHours
                # row[idxFluidRatInUseWash] = float(nowInUseF)/realWT
                # ???????????????
                if changeRate <= 2:
                    targetF = 0
                else:
                    targetF = 1
                    cNotChosen[now] += 1
            else:  # ???????????????????????????????????????
                row[idxFluidRatMeasureWash:idxFluidRatMeasureWash + 3] = 0

            # ??????????????????
            if matchSample:
                subS = idx - preS
                preS = idx

                # ?????????(?????????)
                if matchSample != None:  #
                    nowInUseW = float(row[idxWaterCutInUseWash])  # ???????????????
                    nowMeasureW = float(row[idxWaterCutSampleWash])  # ???????????????
                    changeRate = abs(float(nowInUseW) - float(nowMeasureW))  # ??????

                    # ????????????
                    if changeRate <= 2:
                        targetW = 0
                    else:
                        targetW = 1
            # else:
            # row[idxWaterCutInUseWash] = 0

            labels = np.array([targetF, 1 - targetF])
            labelFormatted.append(labels.astype(np.float))

            # ??????????????????
            '''stopTime, subM, cNotChosenPre, openWell'''
            addInfo.append([stopTime, subM, cNotChosen[pre], openWell])
            # addInfo.append([stopTime, subM, openWell])
            # ??????????????????
            selectedData.append(row)
            selectedIndex.append(idx)
            if targetF == 0:
                cNotChosen[now] = 0
            idx += 1
            now = 1 - now
            pre = 1 - now

        # ?????? remark ??????????????????????????????????????????
        selectedData = np.delete(selectedData, -1, axis=1)

        # ???????????????????????? nan
        for eachData in selectedData:
            for i in range(len(eachData)):
                if isFloat(eachData[i]) != True:
                    eachData[i] = 'nan'

        selectedData = np.append(selectedData, addInfo, axis=1)
        # ??????????????????
        tempData.feature_names = np.delete(tempData.feature_names, -1)
        addFeatureNames = ['stopTime', 'subMeasuring', 'cNotChosenPre', 'openWell']
        tempData.feature_names = np.append(tempData.feature_names, addFeatureNames)
        self.tempData = tempData
        return selectedData

    def get_all_points(self):
        # Step 0.
        # ??????
        measureStr = self.measureStr
        sampleStr = self.sampleStr
        openWellStr = self.openWellStr

        tempData = copy.deepcopy(self.dataSet)
        # ????????????
        delete_columns = [1, 2, 3, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

        idxFluidRatInUseWash = self.idxFluidRatInUseWash
        idxWaterCutInUseWash = self.idxWaterCutInUseWash
        idxFluidRatMeasureWash = self.idxFluidRatMeasureWash
        idxWaterCutSampleWash = self.idxWaterCutSampleWash
        idxWorkingHours = self.idxWorkingHours

        selectedData = []  # list ???????????????????????????
        labelFormatted = self.targetFormatted = []  # list ?????????????????????(target)
        selectedIndex = self.selectedIndex = []  # ??????2???????????????????????????

        # tempData.data = np.delete(tempData.data, delete_columns, axis=1)
        # tempData.feature_names = np.delete(tempData.feature_names, delete_columns)

        idx = 0
        preM = 0
        preS = 0
        addInfo = []
        cNotChosen = [0, 0]
        now = 0
        pre = 1

        num_opt = 3
        mask_opt = []       # fre choke shut_open
        mask_ms = []        # fluid water_cut tem pressure
        opt_interval = []   #
        ms_interval = []    #

        # Step 1.
        # ???????????????????????????????????????
        for row in tempData.data:
            mask_opt_temp = np.zeros(num_opt)
            if idx < self.numDataItems:
                idx += 1
                continue
            matchMeasure = re.match(measureStr, row[-1], re.M | re.I)  # ???????????????
            matchSample = re.match(sampleStr, row[-1], re.M | re.I)  # ???????????????
            matchOpenWell = re.search(openWellStr, row[-1], re.M | re.I)  # ???????????????

            if matchOpenWell != None:
                openWell = 1
                mask_opt_temp[2] = 1
            else:
                openWell = 0
                mask_opt_temp[2] = 0
            # ?????????????????????row ?????????
            if matchSample == None and matchMeasure == None:
                idx += 1
                continue

            # ????????????????????????
            subM = 0
            subS = 0

            # remark?????? ??????????????????
            stopTime = getStopTime(row[-1])
            Date = getDateInterval(row[0])
            row[0] = Date
            workingHours = float(row[idxWorkingHours])
            realWT = workingHours - stopTime
            dayHours = 24

            targetF = 0
            targetW = 0

            # ???????????????
            if matchMeasure:
                subM = idx - preM
                preM = idx

                # ?????????(????????????)
                nowMeasureF = float(row[idxFluidRatMeasureWash])  # ???????????????
                nowInUseF = float(row[idxFluidRatInUseWash])
                changeRate = abs(float(nowInUseF) / realWT - float(nowMeasureF) / dayHours)  # ??????

                # row[idxFluidRatMeasureWash] = float(nowMeasureF)/dayHours
                # row[idxFluidRatInUseWash] = float(nowInUseF)/realWT
                # ???????????????
                if changeRate <= 2:
                    targetF = 0
                else:
                    targetF = 1
                    cNotChosen[now] += 1
            else:  # ???????????????????????????????????????
                row[idxFluidRatMeasureWash:idxFluidRatMeasureWash + 3] = 0

            # ??????????????????
            if matchSample:
                subS = idx - preS
                preS = idx

                # ?????????(?????????)
                if matchSample != None:  #
                    nowInUseW = float(row[idxWaterCutInUseWash])  # ???????????????
                    nowMeasureW = float(row[idxWaterCutSampleWash])  # ???????????????
                    changeRate = abs(float(nowInUseW) - float(nowMeasureW))  # ??????

                    # ????????????
                    if changeRate <= 2:
                        targetW = 0
                    else:
                        targetW = 1
            # else:
            # row[idxWaterCutInUseWash] = 0

            labels = np.array([targetF, 1 - targetF])
            labelFormatted.append(labels.astype(np.float))

            # ??????????????????
            '''stopTime, subM, cNotChosenPre, openWell'''
            addInfo.append([stopTime, subM, cNotChosen[pre], openWell])
            # addInfo.append([stopTime, subM, openWell])
            # ??????????????????
            selectedData.append(row)
            selectedIndex.append(idx)
            if targetF == 0:
                cNotChosen[now] = 0
            idx += 1
            now = 1 - now
            pre = 1 - now

        # ?????? remark ??????????????????????????????????????????
        selectedData = np.delete(selectedData, -1, axis=1)

        # ???????????????????????? nan
        for eachData in selectedData:
            for i in range(len(eachData)):
                if isFloat(eachData[i]) != True:
                    eachData[i] = 'nan'

        selectedData = np.append(selectedData, addInfo, axis=1)
        # ??????????????????
        tempData.feature_names = np.delete(tempData.feature_names, -1)
        addFeatureNames = ['stopTime', 'subMeasuring', 'cNotChosenPre', 'openWell']
        tempData.feature_names = np.append(tempData.feature_names, addFeatureNames)
        self.tempData = tempData
        return selectedData


    def processes_defaults(self, selectedData):
        # ???????????????
        lenFormatedDataItem = self.lenFormatedDataItem = len(selectedData[0])  # ????????????????????????
        selectedData = np.array(selectedData, dtype=np.float32)
        featuresColMeanArray = np.zeros(lenFormatedDataItem, dtype=float)
        featuresColNumArray = np.zeros(lenFormatedDataItem, dtype=float)
        for row in selectedData:
            for i in range(lenFormatedDataItem):
                addmean = row[i]
                addnum = 1
                if np.isnan(row[i]):
                    addmean = 0
                    addnum = 0
                featuresColMeanArray[i] += float(addmean)
                featuresColNumArray[i] += addnum
        for i in range(lenFormatedDataItem):
            if featuresColNumArray[i] != 0:
                featuresColMeanArray[i] /= featuresColNumArray[i]
            else:
                featuresColMeanArray[i] = 0
        for row in selectedData:
            for i in range(len(row)):
                if np.isnan(row[i]):
                    row[i] = featuresColMeanArray[i]
        return selectedData

    def dataFormatHalfPre(self):
        dataFormatted = self.dataFormatted = []  # ?????????????????????

        selectedData = self.get_sample_points()
        if self.dformat != 1:
            self.processes_defaults(selectedData)

        # Step 2.
        # ?????????????????????????????????????????????n???
        idx = 0
        if self.dformat == 1:
            numDataItems = 0
        else:
            numDataItems = self.numDataItems
        for row in selectedData:
            temprow = copy.deepcopy(row)
            for j in range(idx - numDataItems, idx):
                temprow = np.append(temprow, selectedData[j][:])
            dataFormatted.append(temprow.astype(np.float))
            idx += 1

        # ??????????????????
        tempData = self.tempData
        tfeatureNames = copy.deepcopy(tempData.feature_names)
        for i in range(numDataItems):
            j = 0
            for eachname in tfeatureNames:
                tfeatureNames[j] = eachname+str(i)
                j += 1
            tempData.feature_names = np.append(tempData.feature_names, tfeatureNames)

        self.tempData = tempData

        data_interpolate = self.interpolation_liner(selectedData)

        if_plot = 0
        if if_plot == 1:
            from src.plotlib.pdploter import WellDataPloter
            pdploter = WellDataPloter(data_interpolate, self.tempData.feature_names, self.targetFormatted,
                                      self.well_name)
            pdploter.plot_measure_opt_interpolate()

        return tempData

    def dataFormatHalfNext(self):
        # ?????????????????????????????????????????????
        # idx = list(self.tempData.feature_names).index('cNotChosen')
        # delete_columns = [self.idxFluidRatInUseWash, idx]

        delete_columns = [self.idxFluidRatInUseWash]
        dataFormatted = np.array(self.dataFormatted)
        # dataFormatted = np.delete(dataFormatted, delete_columns, axis=1)
        dataFormatted[:, delete_columns] = 0
        # self.tempData.feature_names = np.delete(self.tempData.feature_names, delete_columns)

        start_idx = 0
        temp_idx = 0
        for eachdata in dataFormatted:
            if eachdata[0] >= 0:
                start_idx = temp_idx
                break
            temp_idx += 1

        if self.dformat == 1:
            rnnBatchSize = self.numDataItems
            rnnData = []
            from collections import deque
            que = deque(maxlen=rnnBatchSize)
            for eachData in dataFormatted:
                que.append(list(eachData))
                if len(que) == rnnBatchSize:
                    rnnData.append(list(que))
            dataFormatted = rnnData
            self.targetFormatted = self.targetFormatted[rnnBatchSize - 1:]

        self.tempData.data = dataFormatted[start_idx:]
        self.tempData.target = self.targetFormatted[start_idx:]
        self.tempData.data = np.array(self.tempData.data)
        self.tempData.target = np.array(self.tempData.target)

        if_plot = 1
        if if_plot == 1:
            from src.plotlib.pdploter import WellDataPloter
            pdploter = WellDataPloter(dataFormatted, self.tempData.feature_names, self.targetFormatted, self.well_name)
            pdploter.plot_measure_opt()

        return self.selectedIndex, self.tempData


    def addFormedData(self):
        dataFormatted = self.dataFormatted
        selectedIndex = self.selectedIndex
        idxFluidRatMeasureWash = self.idxFluidRatMeasureWash
        idxFluidRatInUseWash = self.idxFluidRatInUseWash
        lenFormatedDataItem = self.lenFormatedDataItem
        labelFormatted = self.targetFormatted

        # Step 3.
        # ???????????????????????????????????????
        formedData, formedTarget, formedIndex = formRLData(
            dataFormatted, selectedIndex, idxFluidRatMeasureWash, idxFluidRatInUseWash, lenFormatedDataItem)

        # Step 4.
        # ?????????????????????????????????
        self.selectedIndex = np.append(selectedIndex, formedIndex)
        # + dataFormatted
        self.dataFormatted = np.append(dataFormatted, formedData, axis=0)
        # target
        self.targetFormatted = np.append(labelFormatted, formedTarget, axis=0)
        return

    def getNameIndex(self, findName):
        name_list = list(self.tempData.feature_names)
        idx = name_list.index(findName)
        return idx

    def interpolation_liner(self, selectedData):
        orig_data = selectedData
        orig_target = self.tempData.target

        getNameIndex = self.getNameIndex
        fl_idx = getNameIndex('Production (measured)')
        fl_idx_r = getNameIndex('Production (Reduced)')
        openWell_idx = getNameIndex('openWell')

        interpolated_data = []
        len_orig_data = len(orig_data)

        for i in range(len_orig_data - 1):
            row = copy.deepcopy(orig_data[i])
            now_measure = float(orig_data[i, fl_idx])
            next_measure = float(orig_data[i + 1, fl_idx])
            now_reduced = float(orig_data[i, fl_idx_r])
            next_reduced = float(orig_data[i + 1, fl_idx_r])
            sub_measuring = int(orig_data[i + 1, 0]) - int(orig_data[i, 0])

            interpolated_data.append(np.array(row))
            for j in range(1, sub_measuring):
                row[fl_idx] = now_measure + (next_measure - now_measure) * (j) / sub_measuring
                row[fl_idx_r] = now_reduced + (next_reduced - now_reduced) * (j) / sub_measuring
                row[0] = int(row[0]) + 1
                interpolated_data.append(np.array(row))
        interpolated_data.append(orig_data[-1])

        interpolated_data = np.array(interpolated_data)
        return interpolated_data

    def getDataset_interpolate(self):
        # ?????????????????????????????????????????????
        selectedData = self.get_sample_points()
        self.processes_defaults(selectedData)
        return selectedData


    def addNewData(self, data, target):
        return

    def formatNewData(self, data):
        # if self.dataFormatted
        return

