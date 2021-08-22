# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:11:54 2021

@author: Maureen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from arch.unitroot import ADF


def plot_prediction(title_str, series, pred,plot_name='plot.png', plot_fig = True, plot_intervals=False, scale=1.96, 
                    plot_anomalies=False, fig_sz=(17,7)):
    """
        title_str - title for figure
        series - dataframe with timeseries
        pred - predition
        plot_fig - show figure
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
        fig_sz - figure size
    """

    nts = series[pred.index]
    
    rmse = np.sqrt(mean_squared_error(nts, pred))
    
    if plot_fig:
        plt.figure(figsize=fig_sz)
        plt.title(title_str)
    
        if plot_intervals:
            mae = mean_absolute_error(nts, pred) # truth and prediction
            deviation = np.std(nts- pred)
            lower_bond = pred - (mae + scale * deviation)
            lower_bond_df = pd.DataFrame(lower_bond, index = nts.index)
            upper_bond = pred + (mae + scale * deviation)
            upper_bond_df = pd.DataFrame(upper_bond, index = nts.index)
            plt.plot(upper_bond_df, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond_df, "r--")
    
            # Having the intervals, find abnormal values
            if plot_anomalies:
                anomalies = pd.DataFrame(index=nts.index, columns=nts.columns)
                anomalies[nts<lower_bond] = nts[nts<lower_bond]
                anomalies[nts>upper_bond] = nts[nts>upper_bond]
                plt.plot(anomalies, "ro", markersize=10)
            
        plt.plot(nts, label="Actual values")
        plt.plot(pred, "r", label="Prediction")
        
        
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.savefig(plot_name)

    return rmse

'''  old thing '''

def plotHistoricalAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """

    plt.figure(figsize=(15,5))
    plt.title("Historical Average")
    remain_size = len(series) - window
    historical_mean = pd.Series([series[:window].mean()] * remain_size,index=series[window:].index)
    plt.plot(historical_mean, "g", label="Historical Mean")
    rmse = np.sqrt(mean_squared_error(series[window:], historical_mean))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], historical_mean)
        deviation = np.std(series[window:]- historical_mean)
        lower_bond = historical_mean - (mae + scale * deviation)
        upper_bond = historical_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series, label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

    return rmse

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    rmse = np.sqrt(mean_squared_error(series[window:], rolling_mean[window:]))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    return rmse
   
    
    
    
def plotSARIMA(series, window, model, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """

    plt.figure(figsize=(15,5))
    plt.title("SRAIMA")
    #remain_size = len(series) - window
    prediction = model.predict(start=window, end=len(series)-1)
    pred_s = pd.DataFrame(prediction, index = series[window:].index)

    plt.plot(pred_s, "r", label="Prediction")

    rmse = np.sqrt(mean_squared_error(series[window:], prediction))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], prediction) # truth and prediction
        deviation = np.std(series[window:]- prediction)
        lower_bond = prediction - (mae + scale * deviation)
        lower_bond_df = pd.DataFrame(lower_bond, index = series[window:].index)
        upper_bond = prediction + (mae + scale * deviation)
        upper_bond_df = pd.DataFrame(upper_bond, index = series[window:].index)
        plt.plot(upper_bond_df, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond_df, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)


    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

    return rmse


def tsplot(y,sno,lags=None, figsize=(12, 7), style='bmh', plot_fig=True):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    result = sm.tsa.stattools.adfuller(y)
    
    if plot_fig:
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            y.plot(ax=ts_ax)
            
            p_value = result[1]
            ts_ax.set_title('Station{0}, Time Series Analysis Plots\n Dickey-Fuller: p={1:.5f}'.format(sno, p_value))
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.savefig('plot/ts_analysis_02/ts_analysis_'+str(sno).zfill(3)+'.png')
        
    return result
        
        
        
    
