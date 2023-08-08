# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:34:05 2021

@author: K4513621
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from operator import is_not
from functools import partial

##############################################################################
##                            Test for Stationarity
##############################################################################
##
##  1) Augmented Dickey-Fuller(ADF) Test
##
##       Null Hypothesis      (H0): If failed to be rejected, it suggests 
##                                  the time series has a unit root, meaning 
##                                  it is non-stationary. It has some time 
##                                  dependent structure.
##
##       Alternate Hypothesis (H1): The null hypothesis is rejected; it 
##                                  suggests the time series does not have 
##                                  a unit root, meaning it is stationary. 
##                                  It does not have time-dependent structure.
##
##       How to test ADF Hypothesis
##       
##       p-value  > 0.05: Fail to reject the null hypothesis (H0), 
##                        the data has a unit root and is non-stationary.
##
##       p-value <= 0.05: Reject the null hypothesis (H0), 
##                        the data does not have a unit root and is stationary.
##  
##  2) Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
##
##       KPSS is another test for checking the stationarity of a time series. 
##       The null and alternate hypothesis for the KPSS test are opposite that
##       of the ADF test.
##
##       Null Hypothesis          : The process is trend stationary.
##
##       Alternate Hypothesis     : The series has a unit root (series is 
##                                  not stationary).
##
##       p-value  > 0.05: Fail to reject the null hypothesis (H0), 
##                        the series is stationary.
##
##       p-value <= 0.05: Reject the null hypothesis (H0), 
##                        the series is non-stationary.
##
##
##  It is always better to apply both the tests, so that it can be ensured 
##  that the series is truly stationary. Possible outcomes of applying these 
##  stationary tests are as follows:
##
##  Case 1: Both tests conclude that the series is not stationary - 
##          The series is not stationary
##  Case 2: Both tests conclude that the series is stationary - 
##          The series is stationary
##  Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - 
##          The series is trend stationary. Trend needs to be removed 
##          to make series strict stationary. The detrended series is checked 
##          for stationarity.
##  Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - 
##          The series is difference stationary. Differencing is to be used 
##          to make series stationary. The differenced series is checked 
##          for stationarity.
##
##############################################################################

def adf_kpss_test(df):
    adf_column_lst = ['Parent Item Code','ADF Statistic','p-value','Critical Value_1%','Critical Value_5%','Critical Value_10%']
    df_ADF_res = pd.DataFrame(index=np.arange((df.shape[1])-1), columns=adf_column_lst)
    
    kpss_column_lst = ['Parent Item Code','KPSS Statistic','p-value','Critical Value_1%','Critical Value_5%','Critical Value_10%']
    df_KPSS_res = pd.DataFrame(index=np.arange((df.shape[1])-1), columns=kpss_column_lst)

    itemList = list(df.columns)
    itemList.pop(0)
    
    for i in range(0,len(itemList)):

        ## read the time-series as a list
        myList = df[itemList[i]].values
        
        ## remove null or Nan from the list
        myList_no_nan = [x for x in myList if pd.notnull(x)]
        
        # ADF Test
        result = adfuller( myList_no_nan, autolag='AIC')
        
        df_ADF_res['Parent Item Code'][i]   = itemList[i]
        df_ADF_res['ADF Statistic'][i]      = result[0]       
        df_ADF_res['p-value'][i]            = result[1]
        df_ADF_res['Critical Value_1%'][i]  = result[4]['1%']
        df_ADF_res['Critical Value_5%'][i]  = result[4]['5%']
        df_ADF_res['Critical Value_10%'][i] = result[4]['10%']
        
       
        # KPSS Test
        result1 = kpss(myList_no_nan, regression='c', nlags="legacy")

        df_KPSS_res['Parent Item Code'][i]   = itemList[i]
        df_KPSS_res['KPSS Statistic'][i]      = result1[0]       
        df_KPSS_res['p-value'][i]            = result1[1]
        df_KPSS_res['Critical Value_1%'][i]  = result1[3]['1%']
        df_KPSS_res['Critical Value_5%'][i]  = result1[3]['5%']
        df_KPSS_res['Critical Value_10%'][i] = result1[3]['10%']
        
        
    return(df_ADF_res,df_KPSS_res)


##############################################################################
# Main Code Starts Here
##############################################################################

df_comb1 = pd.read_excel("C:\\PythonScripts_DemandForecasting_23Mar2023\\Demand_07Dec2021.xlsx")
df_comb1['Invoice Date'] = df_comb1['Invoice Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m') + pd.tseries.offsets.MonthEnd())
df_comb1 = df_comb1[df_comb1['Invoice Date']>= '2014-01-31']
df_comb1 = df_comb1[df_comb1['Invoice Date']<= '2021-07-31']
df_comb1.set_index("Invoice Date", drop=True, inplace=True)


df_ADF_res,df_KPSS_res = adf_kpss_test(df_comb1)
        