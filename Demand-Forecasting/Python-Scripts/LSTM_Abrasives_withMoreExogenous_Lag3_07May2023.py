"""
Created on Mon Apr 10 11:27:56 2023

@author: Aditi Kannan
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
# dataset = read_csv('C:\\Users\\0110B9744\\Desktop\\abr_10Apr2023.csv', header=0, index_col=0)
# dataset = dataset[['BC26','BC27']]

df_comb1 = pd.read_excel("C:\\Users\\vidya\\OneDrive\\Desktop\\Demand_Forecasting\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_comb1 = df_comb1[df_comb1['Invoice Date']>= '2014-01-31']
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2018-07-31']
#df_comb1.set_index("Invoice Date", drop=False, inplace=True)

## read IIP data and google index
df_iip = pd.read_excel("C:\\Users\\vidya\\OneDrive\\Desktop\\Demand_Forecasting\\ExternalFactors_15Dec2021_selected.xlsx")
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: x.strftime('%Y-%m'))
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_iip.set_index("Invoice Date", drop=False, inplace=True)

## read special discount rate (SPD_Rate)
df_spd = pd.read_excel("C:\\Users\\vidya\\OneDrive\\Desktop\\Demand_Forecasting\\monthlySpecialDiscountRate_sel.xlsx")
df_spd['Invoice Date'] = df_spd['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_spd = df_spd[df_spd['Invoice Date']>= '2014-01-31']
df_spd = df_spd[df_spd['Invoice Date']<= '2018-07-31']

df_spd.rename(columns = {'BC26':'BC26_spd'}, inplace = True)

# ## read intra-categorical features
# df_intraCatVar = pd.read_csv("C:\\PythonScripts_DemandForecasting_23Mar2023\\IntracategoricalFeatures.csv")

# df_comb1 = df_comb1.merge(df_iip[['Invoice Date','IIP_MFG']], on='Invoice Date', how='left')
# df_comb1 = df_comb1.merge(df_spd[['Invoice Date','BC26_spd']], on='Invoice Date', how='left')

df_comb1.set_index("Invoice Date", drop=False, inplace=True)

# converting 'INH_DATE' column to to_datetime format
df_comb1['Invoice Date'] = pd.to_datetime(df_comb1['Invoice Date'],format='%Y-%m-%d')

df_copy = df_comb1.copy()

#df_copy = df_copy[['Invoice Date', 'BC26', 'BC27','BC26_spd','IIP_MFG']]
df_copy = df_copy[['Invoice Date','BC26', 'BC27']]
df_copy=df_copy.dropna()

#################################################################
df3=df_copy.dropna()
df3.rename(columns = {'Invoice Date':'Date'}, inplace = True)
df3.set_index("Date",drop=True, inplace=True)
df3.head()

##########################################################
dataset = df3.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
df2 = pd. DataFrame(dataset, columns=['BC26', 'BC27'])
df2.index = df3.index
#########################################################


df2.reset_index(drop=False, inplace=True)
lag_features = ["BC27"]
window1 = 3
window2 = 6
window3 = 12

df_rolled_3m = df2[lag_features].rolling(window=window1, min_periods=0)
df_rolled_6m = df2[lag_features].rolling(window=window2, min_periods=0)
df_rolled_12m = df2[lag_features].rolling(window=window3, min_periods=0)

df_mean_3m = df_rolled_3m.mean().shift(1).reset_index().astype(np.float32)
df_mean_6m = df_rolled_6m.mean().shift(1).reset_index().astype(np.float32)
df_mean_12m = df_rolled_12m.mean().shift(1).reset_index().astype(np.float32)

df_std_3m = df_rolled_3m.std().shift(1).reset_index().astype(np.float32)
df_std_6m = df_rolled_6m.std().shift(1).reset_index().astype(np.float32)
df_std_12m = df_rolled_12m.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df2[f"{feature}_mean_lag{window1}"] = df_mean_3m[feature]
    df2[f"{feature}_mean_lag{window2}"] = df_mean_6m[feature]
    df2[f"{feature}_mean_lag{window3}"] = df_mean_12m[feature]
    
    df2[f"{feature}_std_lag{window1}"] = df_std_3m[feature]
    df2[f"{feature}_std_lag{window2}"] = df_std_6m[feature]
    df2[f"{feature}_std_lag{window3}"] = df_std_12m[feature]

df2.fillna(df2.mean(), inplace=True)

df2.set_index("Date", drop=False, inplace=True)
df2.head()

df22 = df2.set_index("Date",drop=True)

#################################################################


# #values = dataset.values
# values = df_copy.values

# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
reframed = series_to_supervised(df22, 3, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[25,26,27,28,29,30,31]], axis=1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = 92
train = values[:n_train_hours, :]
test = values[n_train_hours:113, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=12, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast

#inv_yhat = concatenate(( test_X[:, 0:],yhat), axis=1)

inv_yhat = concatenate((yhat, test_X[:, 1:2]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:2]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


from sklearn.metrics import mean_squared_error
from math import sqrt
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

import pandas as pd
df_res_comp = pd.DataFrame()
df_res_comp['Obs']=inv_y
df_res_comp['Pred']=inv_yhat
df_res_comp[["Obs", "Pred"]].plot(figsize=(14, 7))

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from math import sqrt
import numpy as np
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

r2s = r2_score(inv_y, inv_yhat)
print('Test r2-score: %.3f' % r2s)

# normalised_mean_squared_error
NMSE = mean_squared_error(inv_y, inv_yhat) / (np.sum((inv_y - np.mean(inv_y)) ** 2)/(len(inv_y)-1))
print('Test NMSC: %.3f' % NMSE)



MAPE = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100
print('Test MAPE: %.3f' % MAPE)

#Test MAPE: 100.774
#Test MAPE: 31.024
#Test MAPE: 29.141


