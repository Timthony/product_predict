
# 数据下载地址：https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/data
# 使用方法：建立data文件夹，将下载数据解压存放到data文件夹中

import netCDF4
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn import grid_search, datasets, svm, linear_model
import csv
import netCDF4 as nc
import sklearn.decomposition as deco


SEED = 42 # Random seed to keep consistent

'''
Loads a list of GEFS files merging them into model format.
'''
def load_GEFS_data(directory,files_to_use,file_sub_str):
    for i,f in enumerate(files_to_use):
        if i == 0:
            X = load_GEFS_file(directory,files_to_use[i],file_sub_str)
        else:
            X_new = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            X = np.hstack((X,X_new))
    return X

'''
Loads GEFS file using specified merge technique.
'''
def load_GEFS_file(directory, data_type, file_sub_str):
    print ('loading', data_type)
    path = os.path.join(directory, data_type+file_sub_str)
    X = list(nc.Dataset(path,'r+').variables.values())[-1][:,:,:,3:7,3:13] # get rid of some GEFS points
    #X = X.reshape(X.shape[0],55,4,10) 								 # Reshape to merge sub_models and time_forcasts
    X = np.mean(X,axis=1) 											 # Average models, but not hours
    X = X.reshape(X.shape[0],np.prod(X.shape[1:])) 					 # Reshape into (n_examples,n_features)
    return X

'''
Load csv test/train data splitting out times.
'''
def load_csv_data(path):
    data = np.loadtxt(path, delimiter=',',dtype=float,skiprows=1)
    times = data[:,0].astype(int)
    Y = data[:,1:]
    return times,Y

'''
Saves out to a csv.
Just reads in the example csv and writes out 
over the zeros with the model predictions.
'''
def save_submission(preds,submit_name,data_dir):
    fexample = open(os.path.join(data_dir,'sampleSubmission.csv'))
    fout = open(submit_name,'w')
    fReader = csv.reader(fexample,delimiter=',', skipinitialspace=True)
    fwriter = csv.writer(fout)
    for i,row in enumerate(fReader):
        if i == 0:
            fwriter.writerow(row)
        else:
            row[1:] = preds[i-1]
            fwriter.writerow(row)
    fexample.close()
    fout.close()


def run_random_forest(trainX, trainY, testX):
    """Run a random forest regressor model"""

    init_model = ensemble.RandomForestRegressor()
    parameters = {
        'n_estimators': np.linspace(5, trainX.shape[1], 20).astype(int)
    }
    gridCV = grid_search.GridSearchCV(init_model, parameters, cv=10)

    trainX_split, testX_split, trainY_split, testY_split = train_test_split(trainX, trainY, test_size=500)
    gridCV.fit(trainX_split, trainY_split)

    n_estimators = gridCV.best_params_['n_estimators']
    print(n_estimators)

    model = ensemble.RandomForestRegressor(n_estimators=n_estimators)

    print("Fitting model...")

    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    return predictions


def run_svr(trainX, trainY, testX):
    """Run a support vector regression model"""

    init_model = svm.SVR()
    parameters = {
        'C': np.logspace(-5, 5, 10),
        'gamma': np.logspace(-5, 5, 10),
        'epsilon': np.logspace(-2, 2, 10)
    }
    gridCV = grid_search.GridSearchCV(init_model, parameters, cv=10)

    trainX_split, testX_split, trainY_split, testY_split = train_test_split(trainX, trainY, test_size=500)

    gridCV.fit(trainX_split, trainY_split)

    gamma = gridCV.best_params_['gamma']
    C = gridCV.best_params_['C']
    epsilon = gridCV.best_params_['epsilon']

    print(gamma, C, epsilon)

    model = svm.SVR(C=C, gamma=gamma, epsilon=epsilon)

    print("Fitting model...")

    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    return predictions


def run_ridge(trainX, trainY, testX):
    """Run a Ridge model"""

    model = linear_model.RidgeCV(alphas=np.logspace(-0, 3, 100), cv=5)

    print("Fitting model...")

    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    return predictions


def run_gbr(trainX, trainY, testX):
    """Run a Gradient Bosted Regressor model"""

    parameters = {
        "loss": "lad",
        "n_estimators": 3000,
        "learning_rate": 0.035,
        "max_features": 80,
        "max_depth": 7,
        "subsample": 0.5
    }

    model = ensemble.GradientBoostingRegressor(parameters)

    print("Fitting model...")

    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    return predictions


'''
Get the average mean absolute error for models trained on cv splits
'''
def cv_loop(X, y, model, N):
    MAEs = 0
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=.20, random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        mae = metrics.mean_absolute_error(y_cv,preds)
        print ("MAE (fold %d/%d): %f" % (i + 1, N, mae))
        MAEs += mae
    return MAEs/N

'''
Everything together - print statements describe what's happening
'''
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



    # Gotta pick a scikit-learn model
    model = Ridge(normalize=True) # Normalizing is usually a good idea

    print( 'Finding best regularization value for alpha...')
    alphas = np.logspace(-3,1,8,base=10) # List of alphas to check
    alphas = np.array(( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ))
    maes = []
    for alpha in alphas:
        model.alpha = alpha
        mae = cv_loop(trainX,trainY,model,N)
        maes.append(mae)
        print ('alpha %.4f mae %.4f' % (alpha,mae))
    best_alpha = alphas[np.argmin(maes)]
    print('Best alpha of %s with mean average error of %s' % (best_alpha,np.min(maes)))

    print('Fitting model with best alpha...')
    model.alpha = best_alpha
    model.fit(trainX,trainY)

    print('Loading test data...')
    testX = load_GEFS_data(data_dir, files_to_use, test_sub_str)
    print('Raw test data shape', testX.shape)

    #
    # predictions_rf = run_random_forest(trainX, trainY, testX)
    #
    # predictions_svr = run_svr(trainX, trainY, testX)
    #
    # predictions_ridge = run_ridge(trainX, trainY, testX)
    #
    # predictions_gbr = run_gbr(trainX, trainY, testX)
    #
    # parameters = {
    #     "loss": 'ls',
    #     "n_estimators": 3000,
    #     "learning_rate": 0.035,
    #     "max_features": 80,
    #     "max_depth": 7,
    #     "subsample": 0.5
    # }
    #
    # model = GradientBoostingRegressor(parameters)
    #
    # print("CV loop ", cv_loop(trainX, trainY[:, ], model, 10))


    print('Predicting...')
    preds = model.predict(testX)

    print('Saving to csv...')
    save_submission(preds,submit_name,data_dir)

if __name__ == "__main__":
    args = { 'data_dir':  './data/', # Set to your data directory assumes all data is in there - no nesting
        'N': 5,                      # Amount of CV folds
        'cv_test_size': 0.2,         # Test split size in cv
        'files_to_use': 'all',       # Choices for files_to_use: the string all, or a list of strings corresponding to the unique part of a GEFS filename
        'submit_name': 'submission_mod_whours.csv'
    }
    main(**args)
