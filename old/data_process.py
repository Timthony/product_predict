# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
from netCDF4 import Dataset

SEED = 42

"""读取98个太阳能基站的太阳能辐射量，也就是标签数据"""
def import_csv_data():
    df_train = np.loadtxt('train.csv', delimiter=',', dtype=float, skiprows=1)
    # 读取以后对数据进行融合，最后留下两行，一行时间，一行这一天的总发电量

    return df_train

'''分割时间和太阳能数据'''
def split_times(df_data):
    times = df_data[:, 0].astype(int)
    data = df_data[:, 1:]
    return times, data


def get_all_predictors(path, predictors, postfix):
    """Get all the predicting data for train and test"""
    # 读入所有的预测数据
    for i, predictor in enumerate(predictors):
        if i == 0:
            X = get_predictor(path, predictor, postfix)
        else:
            print("开始处理数据1")
            X_append = get_predictor(path, predictor, postfix)
            X = np.hstack((X, X_append))

    return X

def get_predictor(path, predictor, postfix):
    """Get predicting data for train and test for a sepcific predictor"""

    X = list(Dataset(os.path.join(path, predictor + postfix)).variables.values())[-1][:]
    #X = Dataset(os.path.join(path, predictor + postfix)).variables.values()[-1][:]
    print("开始处理数据2")
    X = X.reshape(X.shape[0], 55, 9, 16)
    X = np.mean(X, axis=1)
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    return X


def main():
    # 预测器
    Predictors = [
        'apcp_sfc',
        'dlwrf_sfc',
        'dswrf_sfc',
        'pres_msl',
        'pwat_eatm',
        'spfh_2m',
        'tcdc_eatm',
        'tcolc_eatm',
        'tmax_2m',
        'tmin_2m',
        'tmp_2m',
        'tmp_sfc',
        'ulwrf_sfc',
        'ulwrf_tatm',
        'uswrf_sfc'
    ]


    train_end = '_latlon_subset_19940101_20071231.nc'
    train_path = 'train/'

    test_end = '_latlon_subset_20080101_20121130.nc'
    test_path = 'test/'

    print("Importing trainX, testX...")
    # 训练数据包括15类天气文件的合成，从1994-2007
    train_x_all = get_all_predictors(train_path, Predictors, train_end)
    # 测试数据包括15类天气文件的合成，从2008-2012
    test_x_all = get_all_predictors(test_path, Predictors, test_end)
    print("Shape of trainX: ", np.shape(train_x_all))
    print("Shaoe of testX: ", np.shape(test_x_all))

    print("Importing trainY...")
    df_train = import_csv_data()
    print("原始的基站数据", np.shape(df_train))
    times, train_y_all = split_times(df_train)
    print("Shape of trainY: ", np.shape(train_y_all))          # 98个基站每一个的发电量
    # 将98个太阳能基站的数据相加，每天只有一个预测总量
    train_y_1 = np.empty([5113, 1])
    for i in range(0, 5113):
        train_y_1[i][0] = np.sum(train_y_all[i])
    print("修改后的基站辐射量总和为", np.shape(train_y_1))
    c = np.column_stack((train_x_all, train_y_1))
    print("增加以后的训练数据维度为：", np.shape(c))

    np.savetxt("train_prc.csv", c, delimiter=',')
    np.savetxt("test_prc.csv", test_x_all, delimiter=',')


if __name__ == "__main__":
    main()













































































