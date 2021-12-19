# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:36:26 2021

@author: Maureen
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
import datetime as dt
import time
from ts_plot import plot_prediction, tsplot
#import threading
#import multiprocessing as mp
#import dask
#from dask.distributed import Client, progress
from concurrent.futures import ProcessPoolExecutor
import ub_config as cfg



def optimizeSARIMA(series, parameters_list, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in parameters_list:
        print('param:',param)
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(series.values, order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], s)).fit(low_memory=True)
        except:
            continue

        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            #best_model = model
            best_aic = aic
            #best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table

def sarima_model_fit(params):
    #print("sarima_model_fit:", params[3],', ',params[4])
    best_model=sm.tsa.statespace.SARIMAX(params[0],order=params[1],trend='c',seasonal_order=params[2])
    best_fit = best_model.fit(disp=-1)
    prediction = best_fit.predict(start=params[3], end=params[4])
    #print("sarima_model_fit:", prediction)
    return prediction

'''
def predict_SARIMA_dask(series, params = [(6,1,0),(0,1,1,24)], startd='20180601 00:00:00', endd='20180630 23:00:00', freq=1):

    futures = []
    if False:
        for parameters in input_params.values:
            future = client.submit(costly_simulation, parameters)
            futures.append(future)
    
    input_params = []
    pred = pd.DataFrame()
    try:
        f_str = str(freq)+'H'
        for pred_date in pd.date_range(start=startd, end=endd,freq = f_str):
            #print('predict_SARIMA:', pred_date, 'freq:',f_str)
            end_date = pred_date - dt.timedelta(hours=1)
            start_date = pred_date - dt.timedelta(days=7)
            pred_end_date = pred_date + dt.timedelta(hours=(freq-1))

            start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
            end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
            pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
            pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
            #print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)

            train_ts = series[start_date_str:end_date_str]
            series = series.asfreq('H')
            param = [train_ts, params[0], params[1],pred_date_str,pred_end_date_str]
            input_params = input_params + [param]
        print('predict_SARIMA: input parameter done')
        for parameters in input_params:
            #print('parameters:',parameters[3])
            future = client.submit(sarima_model_fit, parameters)
            futures.append(future)
            
        print(client)
        print('start to gather data')   
        results = client.gather(futures)
        print('turn to dataframe')
        for x in results:
            pred =  pd.concat([pred, x])
        
    except Exception as e:
        print('Predict_SARIMA Error:', e)

    return pred
'''
def predict_SARIMA_dask2(series, params = [(6,1,0),(0,1,1,24)], startd='20180601 00:00:00', endd='20180630 23:00:00', freq=1):

    futures = []
    '''
    for parameters in input_params.values:
        future = client.submit(costly_simulation, parameters)
        futures.append(future)
    '''
    input_params = []
    pred = pd.DataFrame()
    e = ProcessPoolExecutor(6)
    try:
        f_str = str(freq)+'H'
        for pred_date in pd.date_range(start=startd, end=endd,freq = f_str):
            #print('predict_SARIMA:', pred_date, 'freq:',f_str)
            end_date = pred_date - dt.timedelta(hours=1)
            start_date = pred_date - dt.timedelta(days=7)
            pred_end_date = pred_date + dt.timedelta(hours=(freq-1))

            start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
            end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
            pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
            pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
            #print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)

            train_ts = series[start_date_str:end_date_str]
            series = series.asfreq('H')
            param = [train_ts, params[0], params[1],pred_date_str,pred_end_date_str]
            input_params = input_params + [param]
        print('predict_SARIMA: input parameter done')
        for parameters in input_params:
            #print('parameters:',parameters[3])
            future = e.submit(sarima_model_fit, parameters)
            futures.append(future)
        
        ##print(client)
        print('start to gather data')   
        ##results = client.gather(futures)
        results = [future.result() for future in futures]
        print('turn to dataframe')
        for x in results:
            pred =  pd.concat([pred, x])
        
    except Exception as e:
        print('Predict_SARIMA Error:', e)

    return pred

def predict_SARIMA(series, sno ,params = [(6,1,0),(0,1,1,24)], startd='20180601 00:00:00', endd='20180630 23:00:00', freq=1):

    pred = pd.DataFrame()
    try:
        f_str = str(freq)+'H'
        for pred_date in pd.date_range(start=startd, end=endd,freq = f_str):
            print('predict_SARIMA:', pred_date, 'freq:',f_str)
            end_date = pred_date - dt.timedelta(hours=1)
            start_date = pred_date - dt.timedelta(days=2)
            pred_end_date = pred_date + dt.timedelta(hours=(freq-1))

            start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
            end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
            pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
            pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
            print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)

            train_ts = series[start_date_str:end_date_str]
            series = series.asfreq('H')
            best_model=sm.tsa.statespace.SARIMAX(train_ts,order=params[0],trend='c',seasonal_order=params[1])
            best_fit = best_model.fit(disp=-1)
            prediction = best_fit.predict(start=pred_date_str, end=pred_end_date_str)
            pred = pd.concat([pred, prediction])
            
    except Exception as e:
        print('Predict_SARIMA Error:', e)
        pred.to_csv("error_tmp_"+str(sno).zfill(3)+'.csv')
    return pred

def predict_SARIMA_from_file(sno):

    res_path = "D:/youbike/code/ubike_refactor/v2021/statistic_csv/sarima_preidct_result/"
    fname = "sarima_prediction_" + str(sno).zfill(3) + ".csv"
    df = pd.read_csv(res_path + fname)
    df['time'] = pd.to_datetime(df['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df = df.set_index(df['time']).drop(columns=['Unnamed: 0','time'])
    
    return df

#%%
# Moving Average
def moving_average_result(ts_data, date_list):
    
    #train_start_date = date_list[0]
    #train_end_date = date_list[1]
    test_start_date = date_list[2]
    test_end_date = date_list[3]
    rmse_list = []
    
    for w in [3,6,12]:
        start_date = (dt.datetime.strptime(test_start_date,"%Y%m%d %H:%M:%S") - dt.timedelta(hours=(w))).strftime("%Y%m%d %H:%M:%S")
        ma_y = ts_data[start_date:test_end_date]
        rolling_mean = ma_y.rolling(window=w).mean().shift(1).dropna()
        rmse = plot_prediction("Moving Average window size({})".format(w), ts_data, rolling_mean, plot_fig = False)
        print('Moving Average window({}), rmse:{}'.format(w, rmse))
        rmse_list = rmse_list + [rmse]
    return rmse_list

#%%
#Histroical Average
def historical_average_result(ts_data, date_list):
    
    #train_start_date = date_list[0]
    #train_end_date = date_list[1]
    test_start_date = date_list[2]
    test_end_date = date_list[3]
    
    test_y = ts_data[test_start_date:test_end_date]

    remain_size = len(test_y)
    historical_mean = pd.Series([train_ts.mean()] * remain_size,index=test_y.index)
    rmse = plot_prediction("Historical Average", ts_data, historical_mean, plot_fig = False)
    print('Historical Average, rmse:{}'.format(rmse))
    
    return rmse

#%%
#todo:
import math
def ts_analyze(ts_data, sno):

    found_flag = False
    #print acf and pacf
    res = tsplot(ts_data,sno,lags=60, plot_fig=False)
    d = 0
    while not found_flag:
        if res[0] < res[4]['1%']: #found
            found_flag = True
        else:
            ts_data = ts_data - ts_data.shift(1)
            tsplot(ts_data[1+1:],sno, lags=60, plot_fig=False)
            d = d + 1
            
    if math.frexp(res[1])[1] > (-30):
        ts_data = None
    print('{} p-value: {}'.format(sta,math.frexp(res[1])))
     
    return ts_data
    
#%%
def SARIMA_HyperParam_Training(train_ts):
    
    ps = range(2, 5)
    d =  range(0, 1)
    qs = range(0, 3)
    Ps = range(0, 2)
    D =  range(0, 1)
    Qs = range(0, 2)
    s = 24 # season length is still 24
    
    parameters = product(ps,d,qs, Ps, D, Qs)
    parameters_list = list(parameters)
    result_table = optimizeSARIMA(train_ts, parameters_list,s)
    
    return result_table

#%%
def predict_result(ts_data, arima_params,date_list,station_no,freq_hr=1):
    pred = None
    try:
        print("sno:", station_no)
        test_start_date = date_list[2]
        test_end_date = date_list[3]
        
        res_path = cfg.result_path+"statistic_csv/sarima_preidct_result/" 
        #res_path = "D:/youbike/code/ubike_refactor/v2021/statistic_csv/sarima_preidct_result/"
        ts_data = ts_data.asfreq('H')
        print("sno:", station_no)
        #pred = predict_SARIMA_dask2(ts_data, startd =test_start_date,endd=test_end_date, freq=freq_hr)
        pred = predict_SARIMA(ts_data, station_no,startd =test_start_date,endd=test_end_date, freq=freq_hr)
        #pred =  predict_SARIMA_from_file(station_no)
        print("sno:", station_no)
        #print(pred)
        plot_name_str = res_path + "plot/sarima_prediction_" + str(station_no).zfill(3) +".png"
    
        sarima_title = "SARIMA ({},{},{}) ({},{},{},{}), freq={}h Prediction".format(arima_params[0][0],arima_params[0][1],arima_params[0][2],
                                                                                         arima_params[1][0],arima_params[1][1],arima_params[1][2],arima_params[1][3],1)
        rmse = plot_prediction(sarima_title, ts_data, pred, plot_name=plot_name_str)
        pred.to_csv(res_path + "sarima_prediction_" + str(station_no).zfill(3) +"__.csv")
        return rmse
    
    except Exception as e:
        print("sno:", station_no, " error:", e)
        if pred != None:
            pred.to_csv("sarima_prediction_" + str(station_no).zfill(3) +"_error.csv")
            
            

#%%
''' configuration '''

ha_ma_predict = False
plot_ts_fig = False
arima_hyper_train = False
arima_predict = True

filepath = "E:/csvfile/parsed_2021/2/"
filesinpath = os.listdir(filepath)

train_start_date = '20200714 00:00:00'
train_end_date = '20210131 23:00:00'
test_start_date = '20210201 00:00:00' 
test_end_date = '20210611 18:00:00'
#test_end_date = '20210201 23:00:00'
date_list = (train_start_date,train_end_date,test_start_date,test_end_date)
ignore_list = [15, 20, 160, 198, 199, 200] # no station
ignore_list2 = [28, 47, 58, 69, 99, 101 ,106 ,153 , 168 ,185, 190, 239, 240, 264, 306,311, 313, 378,382,383,387]

station_list = [79]#[31,79,95,124,170,174,237,390]#set(range(369,384)) - set(ignore_list) -set(ignore_list2)
if True:#__name__ == '__main__':
    
    date_list = (train_start_date,train_end_date,test_start_date,test_end_date)
    result_df = pd.DataFrame()
    problem_station = []
    res_time = {}
    for sta in station_list:
        try:

            if False:
                f = "sno_" + str(sta).zfill(3) +"_data.csv"
                print("filename:", f)
                df = pd.read_csv(filepath + f)
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
                df = df.set_index(pd.DatetimeIndex(df['time']))
                df = df.sort_index()
            else:
                f = 'parsed_sno_'+str(sta).zfill(3)+'.csv'
                print("filename:", f) 
                df = pd.read_csv(cfg.csv_parsed_nor_sparsed_db_web_path + f)
                df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
                df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
                
            if plot_ts_fig:
                plt.figure(figsize=((20,7)))
                plt.title("sno {} - Time Series".format(sta))
    
                plt.plot(df.sbi)
                plt.grid(True)
                plt.savefig("sno_"+str(sta).zfill(3)+"_ts_plot.png")
                        
            ts_data = df.sbi #['20200801':'20210531']
            dr = pd.date_range(start='2020-07-14', end='2021-06-11 18:00:00', freq='H')
            #dr = pd.date_range(start='2020-07-14', end='2021-02-01 23:00:00', freq='H')
            ts_data = ts_data.reindex(dr).interpolate(method='linear')

            train_ts = ts_data[train_start_date: train_end_date]
            test_y = ts_data[test_start_date: test_end_date]
            
            #ts_data = ts_analyze(ts_data, sta)
            
            #if not isinstance(ts_data, pd.Series):
            #    problem_station = problem_station + [sta]
            
            if ha_ma_predict:
                res_dict = {}
                res_dict['sno'] = sta
                start_time = time.time() 
                ma_rmse = moving_average_result(ts_data, date_list)
                end_time = time.time()
                res_dict['ma_spendtime'] = end_time - start_time
                start_time = time.time() 
                ha_rmse = historical_average_result(ts_data, date_list)
                end_time = time.time()
                res_dict['ha_spendtime'] = end_time - start_time
                
                res_dict['ha_rmse'] = ha_rmse
                res_dict['ma_rmse_3'] = ma_rmse[0]
                res_dict['ma_rmse_6'] = ma_rmse[1]
                res_dict['ma_rmse_12'] = ma_rmse[2]
                result_df = result_df.append(res_dict, ignore_index=True)
                
            if arima_hyper_train:
                start_time = time.time() 
                result_table = SARIMA_HyperParam_Training(train_ts)
                end_time = time.time()
                optimizeSARIMA_time = (end_time - start_time)
                filename = "sarima_para_tuning_sno{}.csv".format(str(sta).zfill(3))
                para_tuning_df = pd.DataFrame(result_table)
                para_tuning_df.to_csv(filename)
                res_time[sta] = optimizeSARIMA_time
                print('spend time:', optimizeSARIMA_time)
                
            if arima_predict:
                #read parameter from file
                #para_filepath = "D:/youbike/code/ubike_refactor/v2021/statistic_csv/arima_para_tun/"
                para_filepath = "D:/git/youbike_prediction/result/statistic_csv/arima_para_tun/"
                para_filename = "sarima_para_tuning_sno" + str(sta).zfill(3) +".csv"
                para_df = pd.read_csv(para_filepath+para_filename)
                para = para_df.loc[0]['parameters']

                arima_param = [(int(para[1]),int(para[4]),int(para[7])),(int(para[10]),int(para[13]),int(para[16]),24)]
                print('arima_param:',arima_param)
                start_time = time.time() 
                
                rmse = predict_result(ts_data,arima_param,date_list,sta)
                print('rmse:',rmse)
                end_time = time.time()
                optimizeSARIMA_time = (end_time - start_time)
                print('spend time:', optimizeSARIMA_time)              

            
        except Exception as e:
            print(e)
            problem_station = problem_station + [sta]
            
    if ha_ma_predict:
        result_df.to_csv("result/ha_ma_result_0701_finale.csv")
        
    if arima_hyper_train:
        res_t_df = pd.DataFrame(res_time)
        res_t_df.to_csv('result/hyper_train_time.csv')

   ## client.close()

#%%
'''
def thread_job(sta_list):
    
    for sta in sta_list:
        try:
            f = "sno_" + str(sta).zfill(3) +"_data.csv"
            print("filename:", f)
            df = pd.read_csv(filepath + f)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
            df = df.set_index(pd.DatetimeIndex(df['time']))
            df = df.sort_index()
                            
            ts_data = df.sbi
            dr = pd.date_range(start='2020-07-14', end='2021-06-11 18:00:00', freq='H')
            ts_data = ts_data.reindex(dr).interpolate(method='linear')
                
            para_filepath = "D:/youbike/code/ubike_refactor/v2021/statistic_csv/arima_para_tun/"
            para_filename = "sarima_para_tuning_sno" + str(sta).zfill(3) +".csv"
            para_df = pd.read_csv(para_filepath+para_filename)
            para = para_df.loc[0]['parameters']
            arima_param = [(para[0],para[1],para[2]),(para[3],para[4],para[5],24)]
            start_time = time.time() 
                        
            predict_result(ts_data,arima_param,date_list,sta)
            end_time = time.time()
            optimizeSARIMA_time = (end_time - start_time)
            print('spend time:', optimizeSARIMA_time)
            
        except Exception as e:
            print(e)
            

if True:
    station_list_true = list(station_list)
    station_num = len(station_list_true)
    thread_list = []
    sub_num = math.ceil(station_num/20)
    for i in range(0,20):
        print('start thread,'+str(i))
        #print(station_list_true[i*sub_num:i*sub_num+sub_num])
        arg_list = station_list_true[i*sub_num:i*sub_num+sub_num]
        predict_thread = threading.Thread(target = thread_job, args=(arg_list,))
        thread_list = thread_list + [predict_thread]
        predict_thread.start()
        
    for i in range(0,20):
        thread_list[i].join()
        '''
        

#%%
'''
    ps = range(2, 5)
    d =  range(0, 1)
    qs = range(0, 3)
    Ps = range(0, 2)
    D =  range(0, 1)
    Qs = range(0, 2)
    s = 24 # season length is still 24
    
    parameters = product(ps,d,qs, Ps, D, Qs)
    
    type_dict = {}
    cnt = 0
    for i in parameters:
        type_dict[i] = cnt
        cnt = cnt + 1
    print(type_dict)
    
    type_dict2 = {}
    cnt = 0
    parameters = product(ps,d,qs, Ps, D, Qs)
    for i in parameters:
        type_dict2[cnt] = i
        cnt = cnt + 1        
    print(type_dict2)
    
#%%
    filepath = "D:/youbike/code/ubike_refactor/v2021/type_arima_para_tun/"
    filesinpath = os.listdir(filepath)
    output = pd.DataFrame()
    for f in filesinpath:
       type_df = pd.read_csv(filepath + f)
       print(f)

       import re

       dic = {}
       x = re.findall('[0-9]+', f)
       dic['station'] = int(x[0])
       dic[1] = type_df['type'][0]
       dic[2] = type_df['type'][1]
       dic[3] = type_df['type'][2]
       dic[4] = type_df['type'][3]
       dic[5] = type_df['type'][4]
       
       output = output.append(dic, ignore_index=True)
       
    #output.to_csv("arima_type_result.csv")     
    output = output.set_index(output['station'])
    output = output.drop(columns=['station'])
#%%
    output = pd.read_csv("D:/youbike/code/ubike_refactor/v2021/arima_type_result.csv")    
    
#%%
    print(output.columns)
    #output_dict = {}
    output_list = []
    for item in output.iterrows():
        #list_content = ()
        list_content = tuple(sorted(set([item[1][1]])))#, item[1][2]])))#, item[1][3]])))
        #output_dict[item[0]] = list_content
        
        output_list = output_list + [list_content]
       ''' 
