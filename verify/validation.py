import torch
from src.preprocessing.normalizerTensor import tensorNormalizer
from src.loss.lossFunction import py_sigmoid_focal_loss
import torch.utils.data as Data
from abandoned_code.sNetwork import sANN
import numpy as np
from logRmse import log_rmse_batch
from abandoned_code.sNetwork import train

# from trainModel import pre

def getKfoldData(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为测试数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat([y_train, y_part], dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_test, y_test

import sklearn
import matplotlib.pyplot as plt


def getF1scores(target, pred, thresholds=None):
    if thresholds.any() == None:
        thresholds = np.sort(pred)[::-1]
    f1scores = []
    allTP = np.sum(target)
    for eachthreshold in thresholds:
        y_pred = (pred >= eachthreshold)
        right = (y_pred == target)
        TP = 0
        for (eachp, eachr) in zip(y_pred, right):
            if eachp and eachr:
                TP += 1
        predP = np.sum(y_pred)
        if predP == 0:
            P = 0
        else:
            P = TP / np.sum(y_pred)
        R = TP / allTP
        if (P + R) == 0:
            temp_y_F1 = 0
        else:
            temp_y_F1 = 2.0 * P * R / (P + R)
        # f1scores.append(sklearn.metrics.f1_score(target, y_pred))
        f1scores.append(temp_y_F1)

    return np.array(f1scores), thresholds


from tqdm import tqdm


def kFold(k, X, y, num_epochs=401, modelwork=sANN,
          lossFunction=py_sigmoid_focal_loss, learning_rate=0.001,
          weight_decay=1e-3, batch_size=32):
    """
    nInput = len(X[0][0])
    """
    nInput = X.shape[-1]
    nOutput = len(y[0])

    for i in range(k):
        if i < k - 1:
            continue
        X_train, y_train, X_test, y_test = getKfoldData(k, i, X, y)  # 获取k折交叉验证的训练和验证数据)
        normalizer = tensorNormalizer()
        X_train = normalizer.normalization(X_train)
        X_test = normalizer.normalization(X_test)

        model = modelwork(nInput, nOutput, batch_size) # 实例化模型
        # model = sLSTM(nInput, nOutput, batch_size=batch_size)
        # lossFunction = torch.nn.functional.binary_cross_entropy
        lossFunction = py_sigmoid_focal_loss

        """
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        """

        trainData = Data.TensorDataset(X_train, y_train)
        testData = Data.TensorDataset(X_test, y_test)

        trainDataLoader = Data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, drop_last=True)
        testDataLoader = Data.DataLoader(dataset=testData, batch_size=batch_size, drop_last=True)

        train_ls = []
        valid_ls = []
        roundLearingRateArray = [1e-4, 5e-5, 2e-5]
        # model = model.to(device)
        for epoch in tqdm(range(num_epochs), desc='training', ncols=100):
            roundLearingRate = roundLearingRateArray[int(epoch * len(roundLearingRateArray)
                                                         / (num_epochs + 1))]
            optimizer = torch.optim.Adam(model.parameters(), lr=roundLearingRate, weight_decay=weight_decay)
            train(epoch, trainDataLoader, model, lossFunction, optimizer)
            #trloss = log_rmse(1, model, X_train, y_train, lossFunction)
            trloss = log_rmse_batch(1, model, trainDataLoader, lossFunction)
            #teloss = log_rmse(1, model, X_test, y_test, lossFunction)
            teloss = log_rmse_batch(1, model, testDataLoader, lossFunction)
            train_ls.append(trloss)
            valid_ls.append(teloss)

        model.eval()
        label_y1 = None
        pred = None
        for step, (seq, target) in enumerate(testDataLoader):
            if label_y1 is None:
                label_y1 = target
            else:
                label_y1 = torch.cat([label_y1, target], dim=0)
            with torch.no_grad():
                y_pred = model(seq)
                if pred is None:
                    pred = y_pred
                else:
                    pred = torch.cat([pred, y_pred], dim=0)

        label_y1 = 1 - torch.max(label_y1, 1)[1].data.numpy()
        output = pred.detach().numpy()[:, 0]

        fpr1, tpr1, thresholds1 = sklearn.metrics.roc_curve(label_y1, output,
                                                            pos_label=None,
                                                            sample_weight=None,
                                                            drop_intermediate=True)
        plt.figure(11)
        plt.plot(fpr1, tpr1, marker='.', color='b', label='tpr')
        f1scores, thresholdsf1 = getF1scores(label_y1, output, thresholds1)
        # thresholdsf1 = 1 - thresholdsf1
        plt.plot(fpr1, f1scores, marker='.', color='r', label='f1-scores')
        plt.xlabel('fpr')
        plt.legend(loc=0)
        plt.show()

        output = 1 - torch.max(pred, 1)[1].data.numpy()

        idxmax = f1scores.argmax()
        bestthreshold = thresholdsf1[idxmax]
        predb = (output >= bestthreshold)
        right = (predb == label_y1)
        print('best threshold: %.4f' %bestthreshold,
              'best f1-score: %.4f' %(f1scores[idxmax]),
              'bprecision: %.4f' % (sklearn.metrics.precision_score(label_y1, predb)),
              'brecall: %.4f' % (sklearn.metrics.recall_score(label_y1, predb)),
              'bacc: %.4f' %(np.sum(right)/ len(right)))

        plt.figure(10)
        pltx = np.arange(0, num_epochs, 1, dtype=np.int16)
        tloss = np.array(train_ls)[:, 0]
        vloss = np.array(valid_ls)[:, 0]
        plt.plot(pltx, tloss, color='red', label='train loss')
        plt.plot(pltx, vloss, color='blue', label='test loss')
        plt.legend(loc=0)
        plt.show()

        print('*' * 10, 'fold', i + 1, '*' * 10)
        print('train loss:%.6f' % train_ls[-1][0], 'train acc:%.4f\n' % train_ls[-1][1],
              'valid loss:%.6f' % valid_ls[-1][0], 'valid acc:%.4f' % valid_ls[-1][1])
        # pre(model, testDataLoader)


        # pre(model, testDataLoader)
        print('precision: %.4f' % (sklearn.metrics.precision_score(label_y1, output)),
              ' recall: %.4f' % (sklearn.metrics.recall_score(label_y1, output)),
              ' f1score: %.4f' % (sklearn.metrics.f1_score(label_y1, output)))
        print('truePosLabel: %.4f' % (np.sum(label_y1) / len(label_y1)),
              ' predPosLabel: %.4f' % (np.sum(output) / len(label_y1)))

        model.train()
