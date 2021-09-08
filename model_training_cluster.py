import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys, time
import datetime as dt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_classif,f_regression
from sklearn.feature_selection import SelectKBest

import ub_config as cfg
from utility import data_preprocess

# model
import baseline
from ridge_model import ridge
from lasso_model import lasso
from gb_model import gb
from ada_model import ada
from rf_model import rf
from xgb_model import xgb

from sklearn.cluster import KMeans
from  sklearn.metrics import silhouette_score
    
''' configuration '''
small_set_flag = False

#estimator_list = [lasso, ridge, rf]
#title_list = ['Lasso ','Ridge ','Random Forest ']

estimator_list = [lasso]
title_list = ['Lasso ']


ignore_list = [15, 20, 160, 198, 199, 200] # no station
ignore_list2 = [28, 47, 58, 69, 99, 101 ,106 ,153 , 168 ,185, 190, 239, 240,264,306,311, 313, 378,382,383,387]

#station_list = set(range(51,405)) - set(ignore_list) -set(ignore_list2)
#station_list = set(range(1,405)) - set(ignore_list) -set(ignore_list2)
station_list = [1,2,3]

filepath = cfg.csv_parsed_db_web_path
station_info_file = cfg.ipython_path + "youbike_station_info.csv"

''' functions '''
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

read_from_parsed = False 

def read_data(sno):
    if read_from_parsed:
        f = 'parsed_sno_'+str(sno).zfill(3)+'.csv'
        print("filename:", f)
        df = pd.read_csv(cfg.csv_parsed_db_web_path + f)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
        
        
    else:
        f = 'merged_sno_'+str(sno).zfill(3)+'_data.csv'
        print("filename:", f)
        df = pd.read_csv(cfg.csv_merged_db_web_path + f)
        df = data_preprocess(df)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
    
    df = df.dropna()
    return df

''' find clusters'''
one_station_one_cluster = True
all_station_one_cluster = False
some_station_one_cluster = False
cluster_num = 0
cluster_dict = {}

if one_station_one_cluster:
    cluster_num = len(station_list)
    for i in range(0,cluster_num):
        cluster_dict[i] = [station_list[i]]
 
if all_station_one_cluster:
    cluster_num = 1
    cluster_dict[0] = station_list
 

if some_station_one_cluster:
    station_info = pd.read_csv(station_info_file)
    #print(station_info.head())
    X = station_info['coordinate'].apply(eval)
    X = X.to_list()
    X = np.asarray(X)
    
    if False: #find best k for cluster
        res_list = []
        for k in range(2,100):
            print('k:',k) #k=28
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            labels = kmeans.labels_
            score = silhouette_score(X, labels, metric='haversine')
            #res = kmeans.predict([[0, 0], [12, 3]])
            res_list = res_list + [score]
            
        plt.plot(res_list)
        print(res_list)
        
    cluster_num = 28
    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(X)
    labels = kmeans.labels_
    for l in labels:
        cluster_dict[l] = []
    for i in range(0,len(labels)):
        l = labels[i]
        #print(station_info.loc[i,:])
        cluster_dict[l] = cluster_dict[l] + [station_info.loc[i,:]['sno']]

print('cluster number:', cluster_num)
for i in range(0,cluster_num):
   print('c',i,': ',cluster_dict[i])
 
#%%   
''' training model '''

train_start_date = '20200521 13:00:00'
train_end_date = '20210131 23:00:00'
test_start_date = '20210201 00:00:00'
test_end_date = '20210611 17:00:00'

for cno, clst in cluster_dict.items():
    
    train_x = pd.DataFrame()
    train_y = pd.DataFrame()
    
    for sno in clst:
        #read from file
        new_df = read_data(sno)
        
        final_drop_list =  ['y_sbi'] 
        x_sta = new_df.drop(columns = final_drop_list)
        y_sta = new_df[['y_sbi']]
        
        sta_train_x, sta_train_y = x_sta[train_start_date:train_end_date], y_sta[train_start_date:train_end_date]
        
        train_x = train_x.append(sta_train_x)
        train_y = train_y.append(sta_train_y)
        
    # Feature Selection
    print('current columns:',train_x.columns)
    print('number of columns:',len(train_x.columns))

    if False:
        corr_thrd = 0.3
        train_corr = train_x.join(train_y['y_sbi'])
        train_corr = train_corr.corr()
        train_corr['y_sbi'].plot(kind="bar",figsize=(20,7)
                                      )
        train_x = train_x[train_corr['y_sbi'][train_corr['y_sbi'].abs() > corr_thrd].index.drop('y_sbi')]
        print(train_x.columns)
            
    if False:
        # 選擇要保留的特徵數
        select_k = 35
        selection = SelectKBest(f_regression, k=select_k).fit(train_x, train_y)
            
        # 顯示保留的欄位
        features = train_x.columns[selection.get_support()]
        #print(features)
        train_x = train_x[features]
        
    
        #get data without datetime index
    train_x_wo_t = train_x.reset_index().drop(columns=['time'])
    train_y_wo_t = train_y.reset_index().drop(columns=['time'])

            
    for i, estimator in enumerate(estimator_list):
        print("current estimator:", title_list[i])
        res = {}
    
        # start training model
        start_time = time.time()
        indexing = None
        model, results, best_param, indexing = estimator(train_x_wo_t, train_y_wo_t,rfecv_en=False)
        end_time = time.time()     
        
        for sno in clst:
            new_df = read_data(sno)
            final_drop_list =  ['y_sbi'] 
            x_sta = new_df.drop(columns = final_drop_list)
            y_sta = new_df[['y_sbi']]
            
            test_x, test_y = x_sta[test_start_date:test_end_date], y_sta[test_start_date:test_end_date]
            
            test_x_wo_t = test_x.reset_index().drop(columns=['time'])
            test_y_wo_t = test_y.reset_index().drop(columns=['time'])
            
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
            print(rmse)
        
        
    break

        

