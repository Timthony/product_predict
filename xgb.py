
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

'''
Get the average mean absolute error (MAE) for models trained on CV splits
'''
def cv_loop(X, y, model, N, cv_test_size):
    MAEs = 0
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=cv_test_size, random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        preds = np.clip(preds, np.min(y_train), np.max(y_train))
        mae = metrics.mean_absolute_error(y_cv,preds)
        print("MAE (fold %d/%d): %f" % (i + 1, N, mae))
        MAEs += mae
        return MAEs/N

'''
Saves predictions to csv file suitable for submission to Kaggle.
Reads in the example csv and writes out over the zeros with the model predictions.
'''
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

    print('Finding best values for PCA components and RF number of estimators...')
    pca_ks = [50, 100]		# list of PCA component options to check
    pca_list = [50, 50, 100, 100]
    rfc_estimators = [50, 100]	# list of RF estimators to check
    rfc_list = [50, 100, 50, 100]
    maes = []
    # first normalize the data before PCA
    trainX = (trainX - np.mean(trainX, 0)) / np.std(trainX, 0)
    for pca_k in pca_ks:
        # conduct PCA using the decomposition module from sklearn
        pca = deco.PCA(pca_k)
        Xpca = pca.fit_transform(trainX)
        print('Explained variance (first %d components): %.3f'%(pca_k, sum(pca.explained_variance_ratio_)))
        for rfc_est in rfc_estimators:
            model = RandomForestRegressor(n_estimators=rfc_est)
            mae = cv_loop(Xpca,trainY,model,N,cv_test_size)
            maes.append(mae)
            print('PCA components %s RF estimators %s mae %.4f' % (pca_k,rfc_est,mae))
    best_num_PCAs = pca_list[np.argmin(maes)]
    best_num_RFs = rfc_list[np.argmin(maes)]
    print('Best PCAs %s best RFs %s with mean average error of %s' % (best_num_PCAs,best_num_RFs,np.min(maes)))

    # calculate the best X transformation matrix (for transforming the test data before prediction)
    pca = deco.PCA(best_num_PCAs)
    Xpca = pca.fit_transform(trainX)
    X_to_PCA = np.transpose(trainX).dot(Xpca)

    print('Fitting model...')
    mymodel = RandomForestRegressor(n_estimators=best_num_RFs)
    mymodel.fit(Xpca,trainY)

    print('Loading test data...')
    testX = load_GEFS_data(data_dir,files_to_use,test_sub_str)
    print('Raw test data shape', testX.shape)
    # transform test data using the PCA transformation array derived from the training data
    testXpca = testX.dot(X_to_PCA)
    print('PCA transformed test data shape', testXpca.shape)

    print('Predicting...')
    preds = mymodel.predict(testXpca)







    print('Saving to csv...')
    save_submission(preds, submit_name, data_dir)

if __name__ == "__main__":
    args = { 'data_dir':  './data/',
        'N': 10,
        'cv_test_size': 0.3,
        'files_to_use': 'all',
        'submit_name': 'submission.csv'
    }
    main(**args)



