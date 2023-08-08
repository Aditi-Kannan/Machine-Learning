# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:23:55 2023

@author: 0110B9744
"""
from math import sqrt
from numpy import concatenate
import numpy as np
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

df_comb1 = pd.read_excel("C:\\PythonScripts_DemandForecasting_23Mar2023\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_comb1 = df_comb1[df_comb1['Invoice Date']>= '2014-01-31']
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2018-07-31']
#df_comb1.set_index("Invoice Date", drop=False, inplace=True)

## read IIP data and google index
df_iip = pd.read_excel("C:\\PythonScripts_DemandForecasting_23Mar2023\\ExternalFactors_15Dec2021_selected.xlsx")
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: x.strftime('%Y-%m'))
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_iip.set_index("Invoice Date", drop=False, inplace=True)

## read special discount rate (SPD_Rate)
df_spd = pd.read_excel("C:\\PythonScripts_DemandForecasting_23Mar2023\\monthlySpecialDiscountRate_sel.xlsx")
df_spd['Invoice Date'] = df_spd['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_spd = df_spd[df_spd['Invoice Date']>= '2014-01-31']
df_spd = df_spd[df_spd['Invoice Date']<= '2018-07-31']

df_spd.rename(columns = {'BC26':'BC26_spd'}, inplace = True)

# ## read intra-categorical features
# df_intraCatVar = pd.read_csv("C:\\PythonScripts_DemandForecasting_23Mar2023\\IntracategoricalFeatures.csv")



df_comb1 = df_comb1.merge(df_iip[['Invoice Date','IIP_MFG']], on='Invoice Date', how='left')
df_comb1 = df_comb1.merge(df_spd[['Invoice Date','BC26_spd']], on='Invoice Date', how='left')

#df_comb1.set_index("Invoice Date", drop=False, inplace=True)

# converting 'INH_DATE' column to to_datetime format
#df_comb1['Invoice Date'] = pd.to_datetime(df_comb1['Invoice Date'],format='%Y-%m-%d')

df_copy = df_comb1.copy()

#df_copy = df_copy[['Invoice Date', 'BC26', 'BC27','BC26_spd','IIP_MFG']]
df_copy = df_copy[['Invoice Date','BC26', 'BC27']]
df2=df_copy.dropna()
df2.rename(columns = {'Invoice Date':'Date'}, inplace = True)
df2.set_index("Date",drop=False, inplace=True)
df2.head()


df2.reset_index(drop=True, inplace=True)
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

import seaborn as sns
import matplotlib.pyplot as plt
import pmdarima as pm


sns.pairplot(df2, kind="scatter")
plt.show()

df2.Date = pd.to_datetime(df2.Date, format="%Y-%m-%d")
df2["month"] = df2.Date.dt.month
df2["week"] = df2.Date.dt.week
df2["day"] = df2.Date.dt.day
df2["day_of_week"] = df2.Date.dt.dayofweek
df2.head()


df_train = df2[df2.Date < "2017"]
df_valid = df2[df2.Date >= "2017"]

exogenous_features = ['BC27', 'BC27_mean_lag3', 'BC27_mean_lag6',
                      'BC27_mean_lag12', 'BC27_std_lag3', 'BC27_std_lag6',
                      'BC27_std_lag12','month', 'week', 'day', 'day_of_week']

model = pm.auto_arima(df_train.BC26, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.BC26, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast

df_valid[["BC26", "Forecast_ARIMAX"]].plot(figsize=(14, 7))

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from math import sqrt
# calculate RMSE
rmse = sqrt(mean_squared_error(df_valid["BC26"], df_valid["Forecast_ARIMAX"]))
print('Test RMSE: %.3f' % rmse)

r2s = r2_score(df_valid["BC26"], df_valid["Forecast_ARIMAX"])
print('Test r2-score: %.3f' % r2s)

# normalised_mean_squared_error
NMSE = mean_squared_error(df_valid["BC26"], df_valid["Forecast_ARIMAX"]) / (np.sum((df_valid["BC26"] - np.mean(df_valid["BC26"])) ** 2)/(len(df_valid["BC26"])-1))
print('Test NMSC: %.3f' % NMSE)


MAPE = np.mean(np.abs((df_valid["BC26"] - df_valid["Forecast_ARIMAX"]) / df_valid["BC26"])) * 100
print('Test MAPE: %.3f' % MAPE)

#Test MAPE: 23.165
#Test MAPE: 22.140
## Final
#Test MAPE: 17.746

