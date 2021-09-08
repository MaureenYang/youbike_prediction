import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys, time
import datetime as dt
import statsmodels.api as sm
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import ub_config as cfg

from sklearn.metrics import mean_squared_error, mean_absolute_error
import data_preprocessor as dp


# model
import baseline
from ridge_model import ridge
from lasso_model import lasso
from gb_model import gb
from ada_model import ada
from rf_model import rf
from xgb_model import xgb


'''
    * each model for each station
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    
''' configuration '''
small_set_flag = False


estimator_list = [lasso, ridge, rf, ada, xgb]
title_list = ['Lasso ','Ridge ','Random Forest ','Adaboost ','XGBoost']

estimator_list = [lasso, ridge, rf]
title_list = ['Lasso ','Ridge ','Random Forest ']

#estimator_list = [gb]
#title_list = ['Gradient Boost ']

#estimator_list = [ada]#,xgb]
#title_list = ['Adaboost ']#, 'XGBoost']

#estimator_list = [lasso, ridge]
#title_list = ['Lasso ','Ridge ']

#station_list = range(11,31) #cfg.station_sno_list#[1,2,3,4,5,6,7,8,9,10] #[1,41,81,121,161,201,241,281,321,361] #range(1, 100, 40)
ignore_list = [15, 20, 160, 198, 199, 200] # no station
ignore_list2 = [28, 47, 58, 69, 99, 101 ,106 ,153 , 168 ,185, 190, 239, 240,264,306,311, 313, 378,382,383,387]

#station_list = set(range(51,405)) - set(ignore_list) -set(ignore_list2)
station_list = set(range(1,405)) - set(ignore_list) -set(ignore_list2)
station_list = [1,2,3]

''' const'''
filepath = cfg.csv_parsed_db_web_path
#%%

''' functions '''
def predict_SARIMA(series, params = [(6,1,0),(0,1,1,24)], startd='20180601 00:00:00', endd='20180630 23:00:00', freq=1):

    pred = pd.DataFrame()
    try:
        f_str = str(freq)+'H'
        for pred_date in pd.date_range(start=startd, end=endd,freq = f_str):
            try:
                print('predict_SARIMA:', pred_date, 'freq:',f_str)
                end_date = pred_date - dt.timedelta(hours=1)
                start_date = pred_date - dt.timedelta(days=15)
                pred_end_date = pred_date + dt.timedelta(hours=(freq-1))
    
                start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
                end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
                pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
                pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
                #print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)
    
                train_ts = series[start_date_str:end_date_str]
                best_model=sm.tsa.statespace.SARIMAX(train_ts,order=params[0],trend='c',seasonal_order=params[1],enforce_stationarity=False, enforce_invertibility=False)
                best_fit = best_model.fit(disp=-1)
                prediction = best_fit.predict(start=pred_date_str, end=pred_end_date_str)
                
                pred = pd.concat([pred, prediction])
            except Exception as e:
                print('Predict_SARIMA Error:', e)
                
    except Exception as e:
        print('Predict_SARIMA Error:', e)

    return pred

def plot_prediction(title_str, series, startd, endd, pred, plot_pic=True,save_fig=False, plot_intervals=False, scale=1.96, plot_anomalies=False, fig_sz=(60,7)):
    
    nts = series.loc[pred.index]
    rmse = np.sqrt(mean_squared_error(nts, pred))
    
    if plot_pic:
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
                anomalies = pd.DataFrame(index=series.index, columns=series.columns)
                anomalies[series<lower_bond] = series[series<lower_bond]
                anomalies[series>upper_bond] = series[series>upper_bond]
                plt.plot(anomalies, "ro", markersize=10)
                         
        plt.plot(pred, "r", label="Prediction")
        plt.plot(nts, label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        
    if save_fig:
        plt.savefig(cfg.result_plot_pred_path+"all_feature/"+title_str+".png")
    
    return rmse


df_list =[]
rmse_list = []
result_dict = {}
#read data from file
for sno in station_list:
    
    f = 'parsed_sno_'+str(sno).zfill(3)+'.csv'
    print("file name:", f)
    df = pd.read_csv(filepath + f)
    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
    df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time','percet','tot'])
    new_df = df.dropna()
    #print(new_df.columns)

    final_drop_list =  ['y_sbi'] 
    X = new_df.drop(columns = final_drop_list)
    Y = new_df[['y_sbi']]
    result_list = []
    
    #split data
    if small_set_flag:
        train_start_date = '20180301 00:00:00'
        train_end_date = '20180531 23:00:00'
        test_start_date = '20180601 00:00:00'
        test_end_date = '20180630 23:00:00'
    else:
        train_start_date = '20200521 13:00:00'
        train_end_date = '20210131 23:00:00'
        test_start_date = '20210201 00:00:00'
        test_end_date = '20210611 17:00:00'
            
    train_x, train_y = X[train_start_date:train_end_date], Y[train_start_date:train_end_date]
    test_x, test_y = X[test_start_date:test_end_date], Y[test_start_date:test_end_date]
        

    # feature selection
    #  - find correlation, only use abs > threshold
    print('current columns:',train_x.columns)
    print('number of columns:',len(train_x.columns))

    if False:
        corr_thrd = 0.3
        train_corr = train_x.join(train_y['y_sbi'])
        train_corr = train_corr.corr()
        train_corr['y_sbi'].plot(kind="bar",figsize=(20,7)
                                      )
        train_x = train_x[train_corr['y_sbi'][train_corr['y_sbi'].abs() > corr_thrd].index.drop('y_sbi')]
        test_x = test_x[train_corr['y_sbi'][train_corr['y_sbi'].abs() > corr_thrd].index.drop('y_sbi')]
        print(train_x.columns)
            
    if False:
        from sklearn.feature_selection import mutual_info_classif,f_regression
        from sklearn.feature_selection import SelectKBest
            
        # 選擇要保留的特徵數
        select_k = 35
        selection = SelectKBest(f_regression, k=select_k).fit(train_x, train_y)
            
        # 顯示保留的欄位
        features = train_x.columns[selection.get_support()]
        print(features)
        train_x = train_x[features]
        test_x = test_x[features]

    if True:
        #get data without datetime index
        train_x_wo_t = train_x.reset_index().drop(columns=['time'])
        train_y_wo_t = train_y.reset_index().drop(columns=['time'])
        test_x_wo_t = test_x.reset_index().drop(columns=['time'])
        test_y_wo_t = test_y.reset_index().drop(columns=['time'])
            
        for i, estimator in enumerate(estimator_list):
            print("current estimator:", title_list[i])
            res = {}
    
            # start training model
            start_time = time.time()
            indexing = None
            model, results, best_param, indexing = estimator(train_x_wo_t, train_y_wo_t,rfecv_en=False)
            end_time = time.time()     
                
            if indexing is not None:
                sel_col = test_x_wo_t.columns[indexing]
                new_test_x_wo_t = test_x_wo_t[sel_col]           
            else:
                new_test_x_wo_t = test_x_wo_t
                
            if estimator == xgb:
                predict_y_wo_t = model.predict(new_test_x_wo_t.values)
            else:
                predict_y_wo_t = model.predict(new_test_x_wo_t)
                    
            predict_y = pd.DataFrame(predict_y_wo_t, index=test_y.index)
        
            predict_y = predict_y.rename(columns={0: 'y_sbi'})      
            stitle = title_list[i] + 'Prediction ,station(' + str(sno) +'), '+ 'sbi'
            rmse = plot_prediction(stitle,test_y, train_start_date,train_end_date,predict_y,plot_pic=True, save_fig=True)
            
            #save result
            res['model'] = title_list[i]
            res['name'] = "All_Feature"
            res['sno'] = sno
            res['results'] = results
            res['best_param'] = best_param
            res['RMSE'] = rmse
            res['time'] = (end_time-start_time)
                
            print('RMSE:', rmse)   
            print("Spent Time:",end_time-start_time,'sec.')
                
            result_list = result_list + [res]
    '''
    else:
        arima_params=[(2,0,1),(1,1,2,24)]
        ts_data = Y.asfreq('H')
        pred = predict_SARIMA(ts_data, startd=test_start_date,endd=test_end_date, freq=1)
        sarima_title = "SARIMA ({},{},{}) ({},{},{},{}), freq={}h Prediction".format(arima_params[0][0],arima_params[0][1],arima_params[0][2],                                                                                             arima_params[1][0],arima_params[1][1],arima_params[1][2],arima_params[1][3],1)
        rmse = plot_prediction(sarima_title, test_y, train_start_date, train_end_date, pred)
        rmse_list = rmse_list + [rmse]
     '''         

#%%
#for sno in range(1,360):
    try:   
        # sarima
        f = 'sarima_prediction_'+str(sno).zfill(3)+'.csv'
        arima_preidct_result_path = "D:/git/youbike_prediction/result/statistic_csv/sarima_preidct_result/"
        arima_result = pd.read_csv(arima_preidct_result_path + f)
        
        arima_result['time'] = pd.to_datetime(arima_result['Unnamed: 0'], format='%Y-%m-%d %H%M%S', errors='ignore')
        arima_result = arima_result.set_index(pd.DatetimeIndex(arima_result['time'])).drop(columns=['time','Unnamed: 0'])
        arima_result = arima_result[:-1]
        arima_result = arima_result.rename(columns={0: 'y_sbi'})
        stitle = 'SRAIMA Prediction ,station(' + str(sno) +'), '+ 'sbi'
        rmse = plot_prediction(stitle, test_y, train_start_date,train_end_date,arima_result,plot_pic=False, save_fig=False)
        print(rmse)
        result_dict[sno] = rmse
        
    except Exception as e:
        print(e)
        result_dict[sno] = None
        

#%%
print(result_dict)

#%%
if False:
    hama_file = "D:/git/youbike_prediction/result/statistic_csv/ha_ma_result_0701_finale.csv"
    hama_result = pd.read_csv(hama_file)
    #print(hama_result)
    hama_df = hama_result[['ha_rmse', 'ma_rmse_12', 'ma_rmse_3', 'ma_rmse_6','sno']]
    hama_df = hama_df.set_index(hama_df.sno).drop(columns=['sno'])
    hama_df['sarima'] = pd.Series(result_dict)
    print(hama_df)
    
    model_file = "D:/git/youbike_prediction/result/training_result/all_feature_trainning_result_001_228.csv"
    model_result = pd.read_csv(model_file)
    print(model_result)
    
    model_list = model_result['model'].unique()
    for m in model_list:
        a = model_result[model_result['model'] == m]
        aa = a[['sno','RMSE']]
        aa = aa.set_index(a['sno']).drop(columns=['sno'])
        print(aa)
        #break
        hama_df[m] = aa
    
    tot_dic = {}
    for sno in station_list:
        f = 'parsed_sno_'+str(sno).zfill(3)+'.csv'
        print("file name:", f)
        df = pd.read_csv(filepath + f)
        #print(df['tot'][0])
        tot_dic[sno] =df['tot'][0]
    
    hama_df['tot'] = pd.Series(tot_dic)
    print(hama_df)
    
    round_df = (hama_df.apply(round))
    for col in round_df.columns:
        if col == 'tot':
            continue
        round_df[col] = round_df[col]/round_df['tot']*100




    #a = round_df['sarima'].dropna()
    #int_idx = round_df.index.intersection(a.index)
    #round_df2 = round_df.loc(int_idx)
    
    round_df2 = round_df.dropna()
    round_df2 = round_df2[round_df2['sarima'] < 100]
    print(round_df2)
    
    print(round_df2['sarima'].describe())
    
    res_of_mean = {}
    for col in round_df.columns:
        if col == 'tot':
            continue
        res_of_mean[col] = round_df2[col].mean()
       
    rrrr = pd.Series(res_of_mean)
    rrrr.plot(kind="bar",figsize=(20,7))
    
    print(res_of_mean)

#%%
'''
res_df = pd.DataFrame()
for k, v in result_dict.items():
    res_df = res_df.append(v)
 
print(res_df)
'''
    
