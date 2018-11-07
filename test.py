# coding=utf-8
import pandas as pd
data = pd.read_csv('train_prc.csv')
testdata = pd.read_csv('test_prc.csv')
print(testdata.shape)