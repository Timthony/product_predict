# coding=utf-8


from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# scaling function
def scale(train_features,test_features):

    scaler=MinMaxScaler()
    data=pd.concat([train_features,test_features])
    scaled_data=pd.DataFrame(scaler.fit_transform(data),columns=test_features.keys())
    train_features=scaled_data.iloc[0:train_features.shape[0]]
    test_features=scaled_data.iloc[train_features.shape[0]:]

    return train_features,test_features

# result saving function
def save_predictions(IDs,predictions,name='result.csv'):
    predictions=pd.DataFrame(list(zip(map(int,IDs),predictions)))
    predictions.to_csv(name,header=False,index=False,sep=',')

def do_LSTM_Regression():

    print("使用LSTM进行回归预测")


    # load precessed data
    train_data=pd.read_csv('train_all_features_lstm.csv')
    train_features=train_data.drop('发电量',axis=1)
    y=train_data['发电量']
    X_test=pd.read_csv('test_all_features_lstm.csv')


    # select best features for LSTM


    #adapt the data form to LSTM
    X_test=X_test
    X_train=train_features

    X_train,X_test=scale(X_train,X_test)# scaling

    X_train=X_train.values.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test=X_test.values.reshape(X_test.shape[0],1,X_test.shape[1])

    #split the data
    X_train,X_val,y_train,y_val=train_test_split(X_train,y,test_size=0.2,random_state=333)


    #construct LSTM based on keras
    model = Sequential()
    model.add(LSTM(200,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    #model.fit(X_train, y_train, epochs=1100, batch_size=100, validation_data=(X_val, y_val), verbose=2, shuffle=False)
    model=load_model('lstm_model_8805_a8485.h5')# load the history best model to recovery the best result
    #prediction
    y_val_pred=model.predict(X_val)
    y_test_pred=model.predict(X_test)

    rmse=mean_squared_error(y_val,y_val_pred)**0.5
    score=1.0/(1.0+rmse)

    y_test_pred=np.array(y_test_pred).flatten()
    ID=pd.read_csv('public.test.csv')['ID']
    save_predictions(ID,y_test_pred,'result_lstm.csv')
    print('score:',score)


# module test
if __name__=='__main__':
    do_LSTM_Regression()
