# coding=utf-8


import csv
import os
import netCDF4 as nc
import numpy as np
import sklearn.decomposition as deco
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from netCDF4 import Dataset
from sklearn.linear_model import Ridge
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
SEED = 42


'''
Loads a list of GEFS files and merges them into model format.
'''
def load_GEFS_data(directory, files_to_use, file_sub_str):
    for i,f in enumerate(files_to_use):
        if i == 0:
            X = load_GEFS_file(directory, files_to_use[i], file_sub_str)
        else:
            X_new = load_GEFS_file(directory, files_to_use[i], file_sub_str)
            X = np.hstack((X, X_new))
    return X

'''
Loads GEFS file using specified merge technique.
'''
def load_GEFS_file(directory, data_type, file_sub_str):
    print('loading', data_type)
    path = os.path.join(directory, data_type+file_sub_str)

    X = list(nc.Dataset(path,'r+').variables.values())[-1][:]            # 自己修改过
    X = X.reshape(X.shape[0],55,9,16)
    X = np.mean(X, axis=1)
    X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
    return X

'''
Load csv test/train Y data splitting out times.
'''
def load_csv_data(path):
    data = np.loadtxt(path, delimiter=',',dtype=float,skiprows=1)
    times = data[:,0].astype(int)
    Y = data[:,1:]
    return times,Y

def save_submission(preds, submit_name, data_dir):
    fexample = open(os.path.join(data_dir, 'sampleSubmission.csv'))
    fout = open(submit_name, 'w')
    fReader = csv.reader(fexample, delimiter=',', skipinitialspace=True)

    fwriter = csv.writer(fout)
    for i, row in enumerate(fReader):
        if i == 0:
            fwriter.writerow(row)
        else:
            row[1:] = preds[i-1]
            fwriter.writerow(row)
    fexample.close()
    fout.close()


def main(data_dir='./data/', N=10, cv_test_size=0.3, files_to_use='all', submit_name='submission.csv'):
    if files_to_use == 'all':
        files_to_use = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc',
                'ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc',
                'pres_msl','spfh_2m','tcolc_eatm','tmax_2m',
                'tmin_2m','tmp_2m','tmp_sfc']
    train_sub_str = '_latlon_subset_19940101_20071231.nc'
    test_sub_str = '_latlon_subset_20080101_20121130.nc'

    print('Loading training data...')
    trainX = load_GEFS_data(data_dir, files_to_use, train_sub_str)                  # 训练样本
    times, trainY = load_csv_data(os.path.join(data_dir, 'train.csv'))              # 训练样本的目标值
    print('Training data shape', trainX.shape, trainY.shape)

    print('Loading test data...')
    testX = load_GEFS_data(data_dir,files_to_use,test_sub_str)
    print('Raw test data shape', testX.shape)

    train_for_matrix = pd.DataFrame(trainX)
    test_for_matrix = pd.DataFrame(testX)
    train_X = train_for_matrix.as_matrix()
    test_X = test_for_matrix.as_matrix()
    train_Y = pd.DataFrame(trainY)




    print("trainX的shape：", train_X.shape)
    print("trainY的shape：", train_Y.shape)

    gbm1 = xgb.XGBClassifier(silent=1, max_depth=10, n_estimators=25000, learning_rate=0.005)
    gbm1.fit(train_X, train_Y)
    predictions = gbm1.predict(test_X)

    print('Saving to csv...')
    save_submission(predictions, submit_name, data_dir)


if __name__ == "__main__":
    args = { 'data_dir':  './data/',
        'N': 10,
        'cv_test_size': 0.3,
        'files_to_use': 'all',
        'submit_name': 'submission1.csv'
    }
    main(**args)

