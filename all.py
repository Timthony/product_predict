# coding=utf-8


import pandas as pd
import numpy as np

lgb_result=pd.read_csv('lgb_final_result.csv')
ID=pd.read_csv('lgb_final_result.csv')['ID']
xgb_result=pd.read_csv('reuslt_xgb1.csv')
lstm_result=pd.read_csv('result_lstm.csv',header=None)[1]

submission = pd.DataFrame({lgb_result*0.35+xgb_result*0.35+lstm_result*0.3})
submission.to_csv('our_result.csv', index= False)



