# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:16:46 2023

@author: Aditi Kannan
"""

# LSTM for Abrasives
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset
# dataset = read_csv('C:\\Users\\0110B9744\\Desktop\\abr_10Apr2023.csv', header=0, index_col=0)
# dataframe = dataset[['BC26']]

df_comb1 = pd.read_excel("C:\\Users\\Aditi Kannan\\Desktop\\Demand_Forecasting\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2019-12-31']
df_comb1 = df_comb1[['Invoice Date','BC26']]
df_comb1 = df_comb1.dropna()
df_comb1.reset_index(inplace = True, drop = True)
# df_train = df_comb1[df_comb1['Invoice Date']<= '2017-12-31']
# df_test =  df_comb1[df_comb1['Invoice Date']> '2017-12-31']

#df_comb1['BC26'].iloc[133]= 50200
df_comb1.set_index('Invoice Date',inplace = True)
df_comb1.plot()

dataset = df_comb1.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

import pandas as pd
df_res_comp = pd.DataFrame()
trainYY=trainY.transpose()
df_res_comp['Obs']=pd.DataFrame(np.array(trainYY))
df_res_comp['Pred']=pd.DataFrame(np.array(trainPredict))
df_res_comp=df_res_comp.dropna()
df_res_comp[["Obs", "Pred"]].plot(figsize=(14, 7))

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from math import sqrt
import numpy as np
# calculate RMSE
rmse = sqrt(mean_squared_error(df_res_comp.Obs, df_res_comp.Pred))
print('Test RMSE: %.3f' % rmse)

r2s = r2_score(df_res_comp.Obs, df_res_comp.Pred)
print('Test r2-score: %.3f' % r2s)

MAPE = np.mean(np.abs((np.array(df_res_comp.Obs), np.array(df_res_comp.Pred)) / np.array(df_res_comp.Obs))) * 100
print('Test MAPE: %.3f' % MAPE)

df_res_comp1 = pd.DataFrame()
testYY=testY.transpose()
df_res_comp1['Obs']=pd.DataFrame(np.array(testYY))
df_res_comp1['Pred']=pd.DataFrame(np.array(testPredict))
df_res_comp1=df_res_comp1.dropna()
df_res_comp1[["Obs", "Pred"]].plot(figsize=(14, 7))

df_res_comp1=df_res_comp1[df_res_comp1['Obs'] !=0]

# calculate RMSE
rmse = sqrt(mean_squared_error(df_res_comp1.Obs, df_res_comp1.Pred))
print('Test RMSE: %.3f' % rmse)

r2s = r2_score(df_res_comp1.Obs, df_res_comp1.Pred)
print('Test r2-score: %.3f' % r2s)

MAPE = np.mean(np.abs((np.array(df_res_comp1.Obs), np.array(df_res_comp1.Pred))) / np.array(df_res_comp1.Obs)) * 100
print('Test MAPE: %.3f' % MAPE)

#Test MAPE: 97.519
#Test MAPE: 97.257
#Test MAPE: 95.910