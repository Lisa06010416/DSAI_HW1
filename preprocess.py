import pandas as pd
from sklearn import preprocessing
import numpy as np


def readFIle(path, type=1):
    if type == 1:
        dataset = pd.read_csv(path)
        dataset = dataset.astype(float)  # 將data轉type
    return dataset


def normalize(dataset):
    # 日期刪掉年份
    dataset["日期"] = (dataset["日期"] % 10000.)

    # 正規化
    dataset = preprocessing.scale(dataset)
    scaler = preprocessing.StandardScaler().fit(dataset)
    preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset = scaler.transform(dataset)

    return dataset
def normalizeTest(dataset,mean,std):

    # 日期刪掉年份
    dataset["日期"] = (dataset["日期"] % 10000.)
    # print('dataset:')
    # print(dataset)

    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    dataset = dataset - mean
    dataset = dataset / std

    dataset[np.isnan(dataset)] = 1e-9
    dataset[dataset == 0] = 1e-9

    # print("normalize:")
    # print(dataset)
    return dataset.values

def getTraindata(dataset_old, dataset_new,mean,std):
    trainData = []
    target = []
    intput2 = []
    day = 7

    dataset_old_n = normalizeTest(dataset_old,mean,std)
    dataset_new_n = normalizeTest(dataset_new,mean,std)
    for i in range(len(dataset_new)):
        l_index = i - day
        r_index = i + day * 2
        if (l_index) >= 0 and (r_index) <= len(dataset_old):
            if (i + day) <= (len(dataset_new)):
                td = []
                td.extend(dataset_old_n[l_index:r_index, 0:5].flatten())  # 要預測的7天  去年同時間的前一個禮拜到後一個禮拜
                # td.extend(dataset_new_n[l_index:i, 2].flatten())  # 前7天
                td.extend(dataset_new_n[l_index:i, 0].flatten())  # 前7天
                trainData.append(td)

                intput2.append(dataset_old.iloc[i:i + day, 2])

                target.append(dataset_new.iloc[i:i + day, 2])
                # target.append(dataset_new.iloc[i:i+day,1])

    return trainData, target, intput2