import numpy as np

from src.file.readfile import loadCSVDataset
from src.preprocessing.strinfoconvert import match_str
from src.preprocessing.strinfoconvert import getStopTime
from src.preprocessing.strinfoconvert import getDateInterval

from src.file.readfile import load_filenames


def getNameIndex(findName, names):
    name_list = list(names)
    idx = name_list.index(findName)
    return idx


def delete_features(dataset, delete_columns):
    dataset.data = np.delete(dataset.data, delete_columns, axis=1)
    dataset.feature_names = np.delete(dataset.feature_names, delete_columns)
    return dataset.data, dataset.feature_names


class welldataloader:
    def __init__(self, filename):
        self.filename = filename
        self.dataset = self.get_data()
        self.processing_replace_reduced_data()  # data, label
        self.processing_sampling_deltat()  # Deltas
        # self.replace_oil_rate()
        self.processing_mask_deltat()  # deltas
        self.remove_columns()

    def get_data(self):
        self.dataset = loadCSVDataset(self.filename)
        """
        dataSet = Bunch(
            data=data,
            feature_names=feature_names)
        """
        self.date = self.dataset.data[:, 0]
        return self.dataset

    def processing_replace_reduced_data(self):
        chosen_label = []
        str_measure = 'measuring'
        str_sample = 'sampling'

        index_measured = getNameIndex('Fluid Rate0\n(bbl/d)', self.dataset.feature_names)
        index_reduced = getNameIndex('Fluid Rate\n(bbl/d)', self.dataset.feature_names)
        index_watercut_mrd = getNameIndex('Measured\n(%)', self.dataset.feature_names)
        index_workinghours = getNameIndex('Working\nours\n()', self.dataset.feature_names)

        str_stoppump = 'Stop pump'
        str_wellshutin = 'well shut in'

        data_length = len(self.dataset.data)
        data = self.dataset.data
        for i in range(data_length):
            cNotChosen = [-1, -1]
            row = data[i]
            matchMeasure = match_str(str_measure, row[-1])  # 关键字匹配
            matchSample = match_str(str_sample, row[-1])  # 关键字匹配
            if matchMeasure == False:  # 没有产液测量值，直接置为缺省
                row[index_reduced] = 'nan'
            else:  # 有产液量测量值
                measure_Fluidrate = row[index_measured]
                reduce_Fluidrate = row[index_reduced]
                if measure_Fluidrate == reduce_Fluidrate:  # 可直接判断为选用
                    cNotChosen[0] = 0
                else:  # 因为值不同，根据具体情况判断是否选用
                    j = 1
                    while (match_str(str_measure, data[i + j, -1]) == False):
                        if match_str(str_stoppump, data[i + j, -1]) == False and \
                                match_str(str_wellshutin, data[i + j, -1]) == False:  # 没有特殊情况可直接判断为未选用
                            row[index_reduced] = data[i + j, index_reduced]
                            cNotChosen[0] = 1
                            break
                        j += 1
                    if cNotChosen[0] == -1:  # cNotChosen[0] == 1 未选用， == -1 进一步判断是否选用
                        if match_str(str_stoppump, row[-1]) == True:  # 如果是停泵造成的
                            stopTime = getStopTime(row[-1])
                            workingHours = row[index_workinghours]
                            reduce_Fluidrate = float(reduce_Fluidrate) / (float(workingHours) - stopTime) * 24
                            if abs(float(reduce_Fluidrate) - float(measure_Fluidrate)) < 50:  # 判断转为24小时产液量后，看结果是否近似
                                row[index_reduced] = measure_Fluidrate
                                cNotChosen[0] = 0
                            else:
                                cNotChosen[0] = 1
                        else:  # 没有停泵情况， 一定是未选用
                            cNotChosen[0] = 1
            # end if matchMeasure

            if matchSample == False:
                row[index_watercut_mrd + 1] = 'nan'  # reduced water cut
            else:  # 有产液量测量值
                sampled_watercut = row[index_watercut_mrd]
                reduce_watercut = row[index_watercut_mrd + 1]
                if sampled_watercut == reduce_watercut:  # 可直接判断为选用
                    cNotChosen[1] = 0
                else:
                    cNotChosen[1] = 1
            # end if matchSample
            chosen_label.append(cNotChosen)
        # end for i
        self.add_info_label_names = ['fluid_label', 'watercut_label']
        self.add_info_label = chosen_label

        return

    def processing_sampling_deltat(self):
        length = len(self.date)
        Deltas_f = np.zeros(length)
        Deltas_b = np.zeros(length)
        temp_delta_t = 1
        for i in range(length):
            Deltas_f[i] = temp_delta_t
            if sum(self.add_info_label[i]) > -2:
                temp_delta_t = 0
            else:
                temp_delta_t += 1
        temp_delta_t = 0
        for i in range(length - 1, -1, -1):  # backward
            Deltas_b[i] = temp_delta_t
            if sum(self.add_info_label[i]) > -2:
                temp_delta_t = 1
            else:
                temp_delta_t += 1

        self.add_info_Deltas_f = Deltas_f
        self.add_info_Deltas_b = Deltas_b
        return

    def processing_mask_deltat(self):
        data = self.dataset.data
        feature_names = self.dataset.feature_names
        length = len(data)
        num_features = len(feature_names)
        temp_mask = np.zeros(num_features)
        temp_delta_t = np.zeros(num_features)
        mask = np.zeros((length, num_features))
        delta_t_forward = np.zeros((length, num_features))
        delta_t_backward = np.zeros((length, num_features))

        for i in range(length):  # forward process
            row = data[i]
            delta_t_forward[i] = temp_delta_t
            for j in range(num_features):
                if row[j] == 'nan':
                    temp_mask[j] = 0
                    temp_delta_t[j] = temp_delta_t[j] + 1
                else:
                    temp_mask[j] = 1
                    temp_delta_t[j] = 0
            mask[i] = temp_mask

        for i in range(length - 1, -1, -1):  # backward process
            row = data[i]
            delta_t_backward[i] = temp_delta_t
            for j in range(num_features):
                if row[j] == 'nan':
                    temp_delta_t[j] = temp_delta_t[j] + 1
                else:
                    temp_delta_t[j] = 0

        self.add_info_mask = mask
        self.add_info_delta_t_forward = delta_t_forward
        self.add_info_delta_t_backward = delta_t_backward
        return

    def replace_oil_rate(self):
        data = self.dataset.data
        feature_names = self.dataset.feature_names
        chosen_label = self.add_info_label
        length = len(chosen_label)
        index_fluidrate_measured = getNameIndex('Fluid Rate0\n(bbl/d)', feature_names)
        index_fluidrate_reduced = getNameIndex('Fluid Rate\n(bbl/d)', feature_names)
        index_oilrate_measured = getNameIndex('Oil Rate0\n(bbl/d)', feature_names)
        index_oilrate_reduced = getNameIndex('Oil Rate\n(bbl/d)', feature_names)
        index_watercut_reduced = getNameIndex('Dece\n(%)', feature_names)
        for i in range(length):
            row = data[i]
            if sum(chosen_label[i]) >= 0:  # 除开都没有计量的情况
                watercut = isFloat(row[index_watercut_reduced])
                oil_cut = 1 - watercut / 100
                row[index_oilrate_reduced] = isFloat(row[index_fluidrate_reduced]) * oil_cut
                row[index_oilrate_measured] = isFloat(row[index_fluidrate_measured]) * oil_cut
            else:
                row[index_oilrate_reduced] = np.nan
                row[index_oilrate_measured] = np.nan
                self.add_info_mask[i, index_oilrate_reduced] = 0
                self.add_info_mask[i, index_oilrate_measured] = 0
                self.add_info_delta_t_forward[i, index_oilrate_reduced] = \
                    self.add_info_delta_t_forward[i - 1, index_oilrate_reduced] + 1

        for i in range(length - 2, -1, -1):  # backward
            if sum(chosen_label[i]) <= 0:
                self.add_info_delta_t_forward[i, index_oilrate_reduced] = \
                    self.add_info_delta_t_forward[i + 1, index_oilrate_reduced] + 1

    def remove_columns(self):
        feature_names = self.dataset.feature_names
        delete_feature_names = ['Date', 'Alwaa', 'NOC', 'Form.', 'Pump\nDept\n(ft.)', 'Pump\nFlowrate\n(bbl/)',
                                'Working\nours\n()', 'Oil Rate0\n(bbl/d)', 'Gas Rate0\n(Mscf/d)', 'Gas Rate\n(Mscf/d)',
                                'Oil Rate\n(bbl/d)', 'GOR\n(scf/stb)', 'Casing\nPressure\n(psi)',
                                'Back\nPressure\n(psi)', 'Flowing\nPressure\n(psi)', 'ESD\nPre.\n(psi)',
                                'Pi\n(psi)', 'Pd\n(psi)', 'VSD\nCurr.\n（A）', 'Remarks']
        delete_columns = []
        for eachname in delete_feature_names:
            idx = getNameIndex(eachname, feature_names)
            delete_columns.append(idx)
        delete_features(self.dataset, delete_columns)
        self.add_info_mask = np.delete(self.add_info_mask, delete_columns, axis=1)
        self.add_info_delta_t_forward = np.delete(self.add_info_delta_t_forward, delete_columns, axis=1)
        self.add_info_delta_t_backward = np.delete(self.add_info_delta_t_backward, delete_columns, axis=1)

    def add_columns(self):
        return

    def make_data(self):
        return {'data': self.dataset.data, 'mask': self.add_info_mask,
                'DeltaF': self.add_info_delta_t_forward, 'DelatB': self.add_info_delta_t_backward,
                'label': self.add_info_label, 'date': self.date}

    def make_TS_data_onlymeasure(self, seqlength):
        data_dict = self.make_TS_data(seqlength)
        return data_dict

    def make_TS_data(self, seqlength):
        data = self.dataset.data
        chosen_label = self.add_info_label
        mask = self.add_info_mask
        delta_t_forward = self.add_info_delta_t_forward
        delta_t_backward = self.add_info_delta_t_backward
        Deltas_f = self.add_info_Deltas_f
        Deltas_b = self.add_info_Deltas_b
        date = self.date

        rnnData = []
        rnnLabel = []
        rnnDate = []
        rnnDelta = []
        from collections import deque
        quedata = deque(maxlen=seqlength)
        queDelta = deque(maxlen=seqlength)
        for eachData, eachMask, eachmDeltaF, eachmDeltaB, eachDeltaF, eachDeltaB, eachLabel, eachDate in \
                zip(data, mask, delta_t_forward, delta_t_backward, Deltas_f, Deltas_b, chosen_label, date):
            if sum(eachLabel) > -2:  # 选取存在采样（产液、含水）的时间点
                tempsample = [eachData, eachMask, eachmDeltaF, eachmDeltaB]
                quedata.append(tempsample)
                queDelta.append([np.array([eachDeltaF]), np.array([eachDeltaB])])
            else:
                continue
            if len(quedata) == seqlength:
                rnnData.append(list(quedata))
                rnnLabel.append(list(eachLabel))
                rnnDelta.append(list(queDelta))
                rnnDate.append(getDateInterval(eachDate))

        try:
            rnnData = np.array(rnnData).astype(float)
            rnnDelta = np.array(rnnDelta).astype(float)
            seqData = rnnData[:, :, 0, :]
            seqMask = rnnData[:, :, 1, :]
            seqDeltaF = rnnData[:, :, 2, :]
            seqDelatB = rnnData[:, :, 3, :]
            Deltas_f = rnnDelta[:, :, 0]
            Deltas_b = rnnDelta[:, :, 1]
            seqLabel = np.array(rnnLabel).astype(float)
            seqDate = np.array(rnnDate).astype(int)
        except:
            print('error with file: ' + self.filename)
            # rnnData = np.array(rnnData).astype(float)

        return {'seqdata': seqData, 'seqmask': seqMask,
                'seqDeltaF': seqDeltaF, 'seqDelatB': seqDelatB,
                'Deltas_f': Deltas_f, 'Deltas_b': Deltas_b,
                'label': seqLabel, 'date': seqDate}


def isFloat(x):
    try:
        x = float(x)
        return x
    except:
        print("waring, type error")
        return np.nan


class WellGroup:
    wellID = []
    dataProcessers = []
    pFeatures = []
    pDataAll = []

    def __init__(self, dirname, seqlength=0):
        self.seqlength = seqlength
        fileNames = load_filenames(dirname)
        for eachFile in fileNames:
            tempPDP = welldataloader(eachFile)
            self.dataProcessers.append(tempPDP)

    def load_TS_data(self, method='onlymeasure'):
        flag = 0
        for eachPDP in self.dataProcessers:
            if method == 'all':
                seq_dataset = eachPDP.make_TS_data(self.seqlength)
            else:
                seq_dataset = eachPDP.make_TS_data_onlymeasure(self.seqlength)

            if len(seq_dataset['seqdata']) == 0:
                print('remove Well: ' + eachPDP.filename)
                continue
            """
            dict
            {'seqdata': seqData, 'seqmask': seqMask,
             'seqDeltaF': seqDeltaF, 'seqDelatB': seqDelatB,
             'label': seqLabel, 'date': seqDate}
             """
            singleWellData = seq_dataset
            if flag == 0:
                self.pFeatures = eachPDP.dataset.feature_names
                self.pDataAll = singleWellData
                flag = 1
            else:
                for key in seq_dataset:
                    self.pDataAll[key] = np.append(self.pDataAll[key], singleWellData[key], axis=0)

    def addFile(self, addFileName):
        return

    def getDataAll(self):
        return WellGroups.pDataAll, WellGroups.pTargetAll

    def get_ts_wells(self, method='onlymeasure'):
        self.load_TS_data(method='onlymeasure')
        self.pDataAll.update({'featureNames': self.pFeatures})
        return self.pDataAll


if __name__ == '__main__':
    # loader = welldataloader('../../../data/wells/wellsAD4-17-3H.csv')
    # seq_dict = loader.make_TS_data(64)
    wellgroup = WellGroup('../../../data/wells/', seqlength=64)
    dataset = wellgroup.get_ts_wells()
    np.save('../../tempdata/wells_data_mask_onlymeasure.npy', dataset)  # 注意带上后缀名
    pass
