# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:58:01 2021

@author: K4513621
"""

##############################################################################
##                            Test for Homogeneity
##############################################################################
##
##  All Homogeneity test functions have almost similar input parameters. 
##   These are:
##
##          x: a vector (list, numpy array or pandas series) data
##      alpha: significance level (default 0.05)
##        sim: No. of monte carlo simulation for p-value calculation. 
##             (default 20000)
##
##  And all Homogeneity tests return a named tuple which contained:
##
##          h: True (if data is nonhomogeneous) or 
##             False (if data is homogeneous)
##         cp: probable change point location
##          p: p value of the significance test
##  U/T/Q/R/V: test statistic which depends on the test method
##        avg: mean values at before and after the change point
##############################################################################

import pandas as pd
import numpy as np
import pyhomogeneity as hg
import matplotlib.pyplot as plt


def homogeneityTest(df):
    column_lst = ['Parent Item Code','h','Change Point','p-value',
                  'Test Statistic', 'Mean1 Before CP','Mean2 After CP']
    df_res = pd.DataFrame(index=np.arange((df.shape[1])-1), columns=column_lst)
    
    itemList = list(df.columns)
    itemList.pop(0)
    
    for i in range(0,len(itemList)):

        ## read the time-series as a list
        myList = df[itemList[i]].values
        
        ## remove null or Nan from the list
        myList_no_nan = [x for x in myList if pd.notnull(x)]
        
        ## pettitt test for homogeneity
        result = hg.pettitt_test(myList_no_nan)
        
        df_res['Parent Item Code'][i]   = itemList[i]
        df_res['h'][i]                  = result[0]       
        df_res['Change Point'][i]       = result[1]
        df_res['p-value'][i]            = result[2]       
        df_res['Test Statistic'][i]     = result[3]
        df_res['Mean1 Before CP'][i]    = result[4][0]
        df_res['Mean2 After CP'][i]     = result[4][1]
        
        print(i)
            
    return(df_res)    


##############################################################################
# Main Code Starts Here
##############################################################################

df_comb1 = pd.read_excel("C:\\PythonScripts_DemandForecasting_23Mar2023\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
df_comb1 = df_comb1[df_comb1['Invoice Date']>= '2014-01-31']
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2021-07-31']
df_comb1.set_index("Invoice Date", drop=True, inplace=True)

df_homogeneity = homogeneityTest(df_comb1)

##############################################################################

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
 
# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df_comb1.BC26); ax1.set_title('Demand: BC26'); ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(df_comb1.BC26.diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(df_comb1.BC26.diff().diff()); ax3.set_title('2nd Order Differencing')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(df_comb1.BC26, ax=ax1)
plot_acf(df_comb1.BC26.diff().dropna(), ax=ax2)
plot_acf(df_comb1.BC26.diff().diff().dropna(), ax=ax3)


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df_comb1.BC26.dropna())
plot_pacf(df_comb1.BC26.diff().dropna())

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_pacf(df_comb1.BC26, ax=ax1)
plot_pacf(df_comb1.BC26.diff().dropna(), ax=ax2)
plot_pacf(df_comb1.BC26.diff().diff().dropna(), ax=ax3)


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_comb1.BC26, order = (3,0,2))
model_fit = model.fit()
model_fit.summary()


model1 = ARIMA(df_comb1.BC26, order = (1,1,2))
model1_fit = model1.fit()
model1_fit.summary()

