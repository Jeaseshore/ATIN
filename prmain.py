import torch
import torch.optim as optim

import numpy as np

import utils
import argparse
import data_loader


import src.modules.TIATEM as TIATEM
import src.modules.TIATTAIN as TIATTAIN
import src.modules.STRNN as STRNN
import src.modules.T_LSTM as T_LSTM
import src.modules.ATTAIN as ATTAIN
import src.modules.BRITS as BRITS
import src.modules.GRU_D as GRU_D

import time

from src import modules

import sklearn
from sklearn import metrics

from src.plotlib.draw2d import draw_lines
from src.file.writefile import writecsvdata
from utils import seed_torch, getF1scores
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=51)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str)
parser.add_argument('--input_size', type=int, default=6)    # 已经根据数据确定，不用输入该参数
parser.add_argument('--hid_size', type=int, default=16)
parser.add_argument('--impute_weight', type=float, default=1)   # 求损失时，插补的权重
parser.add_argument('--label_weight', type=float, default=1)    # 预测时，标签的权重 （因为计算方式刚好可以将二者放在一个数量级，这两个权重暂未写成代码）
parser.add_argument('--dataset', type=str, default='wells_data_mask_onlymeasure_forwards16.npy')  # 选择不同序列长度的数据集
parser.add_argument('--res', type=bool, default=False)          # 测试结果用
args = parser.parse_args()


def train(model, modelname='AEc'):
    data_dict = np.load('./tempdata/' + args.dataset, allow_pickle=True).item()
    train_set, test_set = data_loader.split_set(data_dict, rate=0.3)
    train_set = data_loader.get_data_dict(train_set, is_fluid=True, is_reduce=False)
    data_iter_train = data_loader.get_loader(train_set, batch_size=32)
    test_set = data_loader.get_data_dict(test_set, is_fluid=True, is_reduce=False)
    data_iter_test = data_loader.get_loader(test_set, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()

        P_loss = 0.0
        I_loss = 0.0
        for idx, data in enumerate(tqdm(data_iter_train, desc='training', ncols=100, mininterval=1)):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            P_loss += ret['lossPred'].item()
            I_loss += ret['lossImputation'].item()
            if not (idx % 50):
                print('\repoch:{} average Pred loss:{:.6f} Imputation loss:{:.6f}' \
                      .format(epoch, P_loss / (idx + 1.0), I_loss / (idx + 1.0)))
        # evaluate(model, data_iter)
        if epoch % 1 == 0 and epoch >= 0:
            model_name = modelname + "{:0>2}".format(str(epoch))
            model_path = './runs/' + args.model + '/' + model_name + '.pt'
            torch.save(model, model_path)
            # threshold(model, data_iter_train, model_name=model_name, save_path='./data_analysis/', mode='train')


def threshold(model, iter, save_path='./data_analysis/', model_name='AEc', mode='test'):
    model.eval()
    labels = []
    pred_scores = []

    # get pred(thresholds) and label
    for idx, data in enumerate(tqdm(iter, desc='evaluating', ncols=100, mininterval=1)):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        threshold = ret['pred'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()

        labels += label.tolist()
        pred_scores += threshold.tolist()

    labels = np.array(labels)[:, 0]
    pred_scores = np.array(pred_scores)[:, 0]

    # ROC
    fpr1, tpr1, thresholds1 = sklearn.metrics.roc_curve(labels, pred_scores,
                                                        pos_label=1,
                                                        sample_weight=None,
                                                        drop_intermediate=True)

    f1scores, acc_scores, thresholdsf1 = getF1scores(labels, pred_scores, thresholds1)

    # get best scores
    length = len(thresholdsf1)
    best_final_score = 0
    best_f1 = 0
    best_acc = 0
    best_threshold = 0
    peak_f1 = 0
    for i in range(length):
        final_score = 2 * f1scores[i] * acc_scores[i] / (f1scores[i] + acc_scores[i])
        f1 = f1scores[i]
        if f1 > peak_f1:
            peak_f1 = f1
        if final_score > best_final_score:
            best_final_score = final_score
            best_f1 = f1scores[i]
            best_acc = acc_scores[i]
            best_threshold = thresholdsf1[i]

    ROC_score = metrics.roc_auc_score(labels, pred_scores)
    print(model_name + ' ' + mode +
          " f1: " + str(best_f1) + " acc: " + str(best_acc) +
          " threshold: " + str(best_threshold) + ' ROC_score: ' + str(ROC_score))

    # draw&write scores
    f1scores = f1scores.reshape((1, length))
    acc_scores = acc_scores.reshape((1, length))
    tpr1 = np.array(tpr1).reshape((1, length))
    fpr1 = np.array(fpr1).reshape((1, length))
    # thresholdsf1 = np.array(thresholdsf1).reshape((1, length))
    score_res = np.concatenate((f1scores, acc_scores), axis=0)
    score_res = np.concatenate((score_res, tpr1), axis=0)

    x = np.concatenate((fpr1, fpr1), axis=0)
    x = np.concatenate((x, fpr1), axis=0)

    draw_lines(x=x, y=score_res, title='', x_axis_names='False positive rate', y_axis_names='Score',
               line_num=3, line_mean=['F1-score', 'Accuracy', 'True positive rate'],
               color=['red', 'blue', 'green'],
               is_save=True, save_as=save_path + mode + "_drawsc_" + model_name)

    res_csv = np.concatenate((score_res, fpr1), axis=0)
    thresholds1 = thresholds1.reshape((1, length))
    res_csv = np.concatenate((res_csv, thresholds1), axis=0)
    res_csv = res_csv.transpose(1, 0)       # f1, acc, tpr, fpr, threshold
    # writecsvdata(res_csv, './data_analysis/' + mode + '_scores_' + model_name + '.csv')

    return best_final_score, best_f1, best_acc, ROC_score, peak_f1 #, best_threshold
        # {'final': best_final_score, 'f1': best_f1, 'acc': best_acc, 'threshold': best_threshold}

def run():
    model = getattr(modules, args.model).Model(args.input_size, args.hid_size, args.impute_weight, args.label_weight)

    modelname = args.model
    if args.res == True:
        modelname = 'res'

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model, modelname=modelname)


def eval_everymodel(seed):
    data_dict = np.load('./tempdata/' + args.dataset, allow_pickle=True).item()
    train_set, test_set = data_loader.split_set(data_dict, rate=0.3)
    test_set = data_loader.get_data_dict(test_set, is_fluid=True, is_reduce=False)
    data_iter_test = data_loader.get_loader(test_set, batch_size=32)

    from src.file.readfile import load_filenames
    file_paths, model_files = load_filenames('./runs/' + args.model)

    scores = []

    for eachmode in model_files:
        model = torch.load('./runs/' + args.model + '/' + eachmode)
        eachmode_name = eachmode.replace('.pt', '')
        eachscores = threshold(model, data_iter_test, model_name=eachmode_name+"drawy", save_path='data_analysis/', mode='test')
        scores.append(eachscores)

    # writecsvdata(scores, './data_analysis/' + 'finaLIATTAIN' + str(round) + '.csv')
    writecsvdata(scores, './data_analysis/' + 'finaL' + args.model + str(seed) + '.csv')

if __name__ == '__main__':
    seeds = []
    for i in range(18):
        seed = int(time.time())    #seeds[i]
        seed_torch(seed)
        run()
        eval_everymodel(seed)
        seeds.append(seed)
    print(seeds)

