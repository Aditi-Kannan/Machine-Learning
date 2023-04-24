# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:54:10 2023

@author: Aditi Kannan
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

import datetime as dt
import datetime
#import pyflux as pf


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels import api as sm
from statsmodels.tsa.arima_model import ARIMA

from numpy import array ## it is better to always import array from numpy not from array

import pmdarima as pm


##############################################################################
# Main Code Starts Here
##############################################################################

df_comb1 = pd.read_excel("C:\\Users\\Aditi Kannan\\Desktop\\Demand_Forecasting\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
df_comb1 = df_comb1[df_comb1['Invoice Date']>= '2014-01-31']
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2021-07-31']
#df_comb1.set_index("Invoice Date", drop=False, inplace=True)

## read IIP data and google index
df_iip = pd.read_excel("C:\\Users\\Aditi Kannan\\Desktop\\Demand_Forecasting\\ExternalFactors_15Dec2021_selected.xlsx")
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: x.strftime('%Y-%m'))
df_iip['Invoice Date'] = df_iip['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
#df_iip.set_index("Invoice Date", drop=False, inplace=True)

## read special discount rate (SPD_Rate)
df_spd = pd.read_excel("C:\\Users\\Aditi Kannan\\Desktop\\Demand_Forecasting\\monthlySpecialDiscountRate_sel.xlsx")
df_spd['Invoice Date'] = df_spd['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
df_spd = df_spd[df_spd['Invoice Date']>= '2014-01-31']
df_spd = df_spd[df_spd['Invoice Date']<= '2021-07-31']

df_spd.rename(columns = {'BC26':'BC26_spd'}, inplace = True)

# ## read intra-categorical features
# df_intraCatVar = pd.read_csv("C:\\PythonScripts_DemandForecasting_23Mar2023\\IntracategoricalFeatures.csv")



df_comb1 = df_comb1.merge(df_iip[['Invoice Date','IIP_MFG']], on='Invoice Date', how='left')
df_comb1 = df_comb1.merge(df_spd[['Invoice Date','BC26_spd']], on='Invoice Date', how='left')

df_comb1.set_index("Invoice Date", drop=False, inplace=True)

# converting 'INH_DATE' column to to_datetime format
df_comb1['Invoice Date'] = pd.to_datetime(df_comb1['Invoice Date'],format='%Y-%m-%d')

df_copy = df_comb1.copy()

df_copy = df_copy[['Invoice Date', 'BC26', 'BC27','BC26_spd','IIP_MFG']]

sns.pairplot(df_copy, kind="scatter")
plt.show()

df_copy[[ 'BC26', 'BC27']].plot(figsize=(14, 7))
df_copy[[ 'BC26_spd']].plot(figsize=(14, 7))
df_copy[[ 'IIP_MFG']].plot(figsize=(14, 7))



df_copy.reset_index(drop=True, inplace=True)

df_copy.set_index("Invoice Date", drop=False, inplace=True)
df_copy.head()

df_copy['Invoice Date'] = pd.to_datetime(df_copy['Invoice Date'], format="%Y-%m-%d")
df_copy["month"] = df_copy['Invoice Date'].dt.month
df_copy.head()


df_train = df_copy[df_copy['Invoice Date'] < "2018"]
df_valid = df_copy[(df_copy['Invoice Date'] >= "2018") & (df_copy['Invoice Date'] <"2019")]

exogenous_features = ["BC27","BC26_spd","IIP_MFG"]

model = pm.auto_arima(df_train.BC26, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.BC26, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features],return_conf_int=True)

df_valid["Forecast_ARIMAX"] = forecast[0]
df_bnd = pd.DataFrame(forecast[1])
df_bnd.rename( columns={0:'LB',1:'UB'}, inplace=True )

df_valid["Forecast_LB"] = list(df_bnd['LB'])
df_valid["Forecast_UB"] = list(df_bnd['UB'])

df_valid[["BC26", "Forecast_ARIMAX","Forecast_LB","Forecast_UB"]].plot(figsize=(14, 7))

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

#Test MAPE: 36.852
