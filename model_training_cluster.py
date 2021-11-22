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
import pickle

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

estimator_list = [lasso, ridge, rf]
title_list = ['Lasso ','Ridge ','Random Forest ']

#estimator_list = [lasso,  ridge]
#title_list = ['Lasso ','Ridge ']

#estimator_list = [lasso ]
#title_list = ['Lasso ']

estimator_list = [ridge ]
title_list = ['Ridge ']

#estimator_list = [rf ]
#title_list = ['Random Forest ']


station_list = [1]#cfg.station_sno_list
#station_list = cfg.small_station_sno_list

filepath = cfg.csv_parsed_db_web_path
station_info_file = cfg.ipython_path + "youbike_station_info.csv"
neighbor_file = cfg.ipython_path + "station_neibor_result.csv"

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

read_from_parsed = True

def read_data(sno,normalized=True):
    if read_from_parsed:
        f = 'parsed_sno_'+str(sno).zfill(3)+'.csv'
        print("filename:", f) 
        df = pd.read_csv(cfg.csv_parsed_nor_sparsed_db_web_path + f)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
        
    else:
        f = 'merged_sno_'+str(sno).zfill(3)+'_data.csv'
        print("filename:", f)
        df = pd.read_csv(cfg.csv_merged_db_web_path + f)
        df = data_preprocess(df,normalize=normalized)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
    
    df = df.dropna()
    return df

def bool2dec(x):
    r = 0
    try:
        
        if x['cat_hospital_500']:
            r = r | 0x01
            
        if x['cat_college_500']:
            r = r | 0x02
            
        if x['cat_train_500']:
            r = r | 0x04
            
        if x['cat_hospital_1000']:
            r = r | 0x08
        
        if x['cat_college_1000']:
            r = r | 0x10
    
        if x['cat_train_1000']:
            r = r | 0x20
        
    except:
        r = None

    return r

#%%
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
 
station_info = pd.read_csv(station_info_file)

if some_station_one_cluster:
    #station_info = pd.read_csv(station_info_file)
    #print(station_info.head())
    station_info = station_info[station_info['sno'].isin(station_list)]
    X = station_info['coordinate'].apply(eval)
    X = X.to_list()
    X = np.asarray(X)
    
    if True: #find best k for cluster
        if True:
            res_list = []
            for k in range(2,round(len(X)/2)):
                #print('k:',k) #k=28
                kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
                labels = kmeans.labels_
                score = silhouette_score(X, labels, metric='haversine')
                #res = kmeans.predict([[0, 0], [12, 3]])
                res_list = res_list + [score]
                
            #plt.plot(res_list)
            #print(res_list)
            
        #cluster_num = 28
        cluster_num = 3
        kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(X)
        labels = kmeans.labels_
        print(station_info)
        for l in range(0,cluster_num):
            cluster_dict[l] = []
            
        for i in range(0,len(labels)):
            l = labels[i]
            print(station_info.iloc[i,:]['sno'])
            cluster_dict[l] = cluster_dict[l] + [station_info.iloc[i,:]['sno']]
        
    else: ##use location information
        neighbor_info = pd.read_csv(neighbor_file)
        neighbor_info = neighbor_info[neighbor_info['sno'].isin(station_list)]
        
        neigh_dec = pd.DataFrame()
        neigh_dec['sno'] = neighbor_info['sno']
        neigh_dec['cluster'] = neighbor_info.apply(bool2dec, axis=1)
        clist = list(set(neigh_dec['cluster'].tolist()))
        cluster_num = len(clist)
        print('clist:', clist)
        for l in range(0,cluster_num):
            i = clist[l]
            cluster_dict[i] = []
            
        for i, x in neigh_dec.iterrows():
            l = x.cluster
            cluster_dict[l] = cluster_dict[l] + [x.sno]   
            

print('cluster number:', cluster_num)
for key, value in cluster_dict.items():
    print('c',key, ':', value)
    
#for i in range(0,cluster_num):
#   print('c',i,': ',cluster_dict[i])
 
#%%   
''' training model '''

train_start_date = '20200521 13:00:00'
train_end_date = '20210131 23:00:00'
test_start_date = '20210201 00:00:00'
test_end_date = '20210611 17:00:00'

all_res_lst = []
lasso_coef_df = pd.DataFrame()
result_df_dict = {'ans' : pd.DataFrame()}

for cno, clst in cluster_dict.items():
    print("start to prepare data for cluster {}: {}".format(cno,clst))
    train_x = pd.DataFrame()
    train_y = pd.DataFrame()
    model_dict = {}
    
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
    
    features = train_x.columns
    
    features = ['sbi', 'PrecpHour', 'UVI', 'td', 'HUMD', 'H_24R', 'PRES',
       'TEMP', 'WDSE', 'comfort', 'WD_E', 'WD_ENE', 'WD_ESE', 'WD_N', 'WD_NE',
       'WD_NNE', 'WD_NNW', 'WD_NW', 'WD_S', 'WD_SE', 'WD_SSE', 'WD_SSW',
       'WD_SW', 'WD_W', 'WD_WNW', 'WD_WSW', 'wkdy_0', 'wkdy_1', 'wkdy_2',
       'wkdy_3', 'wkdy_4', 'wkdy_5', 'wkdy_6', 'hrs_0', 'hrs_1', 'hrs_10',
       'hrs_11', 'hrs_12', 'hrs_13', 'hrs_14', 'hrs_15', 'hrs_16', 'hrs_17',
       'hrs_18', 'hrs_19', 'hrs_2', 'hrs_20', 'hrs_21', 'hrs_22', 'hrs_23',
       'hrs_3', 'hrs_4', 'hrs_5', 'hrs_6', 'hrs_7', 'hrs_8', 'hrs_9',
       'holiday', 'sbi_1h', 'sbi_2h', 'sbi_3h', 'sbi_4h', 'sbi_5h', 'sbi_6h',
       'sbi_7h', 'sbi_8h', 'sbi_9h', 'sbi_10h', 'sbi_11h', 'sbi_12h', 'sbi_1d',
       'sbi_2d', 'sbi_3d', 'sbi_4d', 'sbi_5d', 'sbi_6d', 'sbi_7d','station_id']
   
    train_x = train_x[features]
    
    print('current columns:',train_x.columns)
    print('number of columns:',len(train_x.columns))
    
    if False:
        corr_thrd = 0.5
        if True:
            train_corr = train_x.join(train_y['y_sbi'])
            train_corr = train_corr.corr()
            #print(train_corr)
        else:
            corr_df = pd.read_csv(cfg.ipython_path+"correlation_all.csv")
            train_corr = corr_df.set_index(corr_df["Unnamed: 0"])
            #print(train_corr)

        #train_corr['y_sbi'].plot(kind="bar",figsize=(20,7))
        #print(train_corr['y_sbi'].describe())
        train_x = train_x[train_corr['y_sbi'][train_corr['y_sbi'].abs() > corr_thrd].index.drop('y_sbi')]
        features = train_x.columns
        

    '''
    features = ['sbi','wkdy_0', 'wkdy_1', 'wkdy_2',
       'wkdy_3', 'wkdy_4', 'wkdy_5', 'wkdy_6', 'hrs_0', 'hrs_1', 'hrs_10',
       'hrs_11', 'hrs_12', 'hrs_13', 'hrs_14', 'hrs_15', 'hrs_16', 'hrs_17',
       'hrs_18', 'hrs_19', 'hrs_2', 'hrs_20', 'hrs_21', 'hrs_22', 'hrs_23',
       'hrs_3', 'hrs_4', 'hrs_5', 'hrs_6', 'hrs_7', 'hrs_8', 'hrs_9',
       'holiday']
    '''

    features = ['sbi','hrs_0', 'hrs_1', 'hrs_10',
       'hrs_11', 'hrs_12', 'hrs_13', 'hrs_14', 'hrs_15', 'hrs_16', 'hrs_17',
       'hrs_18', 'hrs_19', 'hrs_2', 'hrs_20', 'hrs_21', 'hrs_22', 'hrs_23',
       'hrs_3', 'hrs_4', 'hrs_5', 'hrs_6', 'hrs_7', 'hrs_8', 'hrs_9']

    features = features + ['HUMD','UVI']

    #features = features + ['station_id']
    #features = features + ['lat', 'lng','tot','cat_hospital_500','cat_college_500','cat_train_500', 'cat_hospital_1000','cat_college_1000','cat_train_1000']
    #features = features + ['lat', 'lng']
    #features = features + ['tot']
    #features = features + ['cat_hospital_500','cat_college_500','cat_train_500']
    #features = features + ['cat_hospital_1000','cat_college_1000','cat_train_1000']
 
    
    if False:
        # 選擇要保留的特徵數
        select_k = 35
        selection = SelectKBest(f_regression, k=select_k).fit(train_x, train_y)
            
        # 顯示保留的欄位
        features = train_x.columns[selection.get_support()]
        train_x = train_x[features]
        
    train_x = train_x[features]

    print('current columns:',train_x.columns)
    print('number of columns:',len(train_x.columns))

    #get data without datetime index
    train_x_wo_t = train_x.reset_index().drop(columns=['time'])
    train_y_wo_t = train_y.reset_index().drop(columns=['time'])
    
    if False:
        if False:
            lasso_model, results, best_param, indexing = lasso(train_x_wo_t, train_y_wo_t,rfecv_en=False)
            label_name = train_x_wo_t.columns
            coef_lab = pd.DataFrame(lasso_model.coef_,index=label_name,columns=[sno])
        else:
            coef_lab = pd.read_csv("lasso_coef.csv")
            coef_lab = coef_lab.set_index(coef_lab['Unnamed: 0']).drop(['Unnamed: 0'],axis=1)

        c = 0.1
        features = coef_lab[coef_lab.abs() > c].dropna().index
        print(coef_lab)
        
        print('c:',c)
        print('n of feature:',len(features))
        print(features)          
        
        train_x_wo_t = train_x_wo_t[features]
    
    
    print("start to training cluster{}: {}".format(cno, clst))
    
    for i, estimator in enumerate(estimator_list):
        print("current estimator{}:{}".format(i,title_list[i]))
        
        # start training model
        start_time = time.time()
        indexing = None
        model, results, best_param, indexing = estimator(train_x_wo_t, train_y_wo_t,rfecv_en=False)
        end_time = time.time()    
        
        #label_name = train_x_wo_t.columns  
        #coef_lab = pd.DataFrame(model.coef_,index=label_name,columns=[sno])
        #lasso_coef_df = lasso_coef_df.append(coef_lab.T)
        #coef_lab = coef_lab.sort_values(by=sno)
        #label_name = coef_lab.index
        #num_feature = len(model.coef_)
        #plt.figure(figsize=(60,7))
        #plt.bar(range(1,num_feature*4,4), coef_lab[0])
        #plt.xticks(range(1,num_feature*4,4), label_name)
        #plt.title("Lasso Feature Importances")
        #plt.show()
        
        res_sno_dict = {}
        for sno in clst:
            new_df = read_data(sno)
            
            final_drop_list =  ['y_sbi'] 
            x_sta = new_df.drop(columns = final_drop_list)
            y_sta = new_df[['y_sbi']]
            
            test_x, test_y = x_sta[test_start_date:test_end_date], y_sta[test_start_date:test_end_date]
            test_x = test_x[features]
            test_x_wo_t = test_x.reset_index().drop(columns=['time'])
            test_y_wo_t = test_y.reset_index().drop(columns=['time'])
            
            if indexing is not None:
                sel_col = test_x_wo_t.columns[indexing]
                new_test_x_wo_t = test_x_wo_t[sel_col]           
            else:
                new_test_x_wo_t = test_x_wo_t
                
            print((new_test_x_wo_t.columns))
                    
            if estimator == xgb:
                predict_y_wo_t = model.predict(new_test_x_wo_t.values)
            else:
                predict_y_wo_t = model.predict(new_test_x_wo_t)
            
            predict_y = pd.DataFrame(predict_y_wo_t, index=test_y.index)
            
            #pickle_file_name = "pickle/model_ridge_sno_" + str(sno).zfill(4) + ".pickle"
            #with open(pickle_file_name, 'wb') as handle:
            #    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)            
            
            predict_y = predict_y.rename(columns={0: 'y_sbi'})
            stitle = title_list[i] + 'Prediction ,station(' + str(sno) +'), '+ 'sbi'
            rmse = plot_prediction(stitle,test_y, train_start_date,train_end_date,predict_y,plot_pic=False, save_fig=False)
            tot = station_info[station_info['sno'] == sno].tot
            res_sno_dict[sno] = [rmse, (rmse/tot)]
            
            result_df_dict['ans']['y_sbi_' + str(sno).zfill(3)] = test_y.loc[predict_y.index]

            
        res_dict ={}
        model_dict = {}
        res_dict['train_time'] = end_time - start_time
        model_dict['name'] = title_list[i]
        model_dict['cluster'] = {'cno':cno,'clist':clst}
        res_dict['model'] = model_dict
        res_dict['pred_rmse'] = res_sno_dict
                
        all_res_lst = all_res_lst + [res_dict]
        
#%%
if False:
    for i, estimator in enumerate(estimator_list):
        result_df_dict[title_list[i]].to_csv("result_" + title_list[i] +".csv")

result_df_dict['ans'].to_csv("result_answer.csv")

#%%
# for small station, find the average RMSE
if True:
    for i, estimator in enumerate(estimator_list):
        small_rmse = [0,0]
        mid_rmse = [0,0]
        big_rmse = [0,0]
        final_time = 0
        for r in all_res_lst:
            if title_list[i] == r['model']['name']:
                #print("model:", title_list[i])
                for s in r['model']['cluster']['clist']:
                    if s in cfg.small_sno:
                        #print('small')
                        small_rmse[0] = small_rmse[0] + r['pred_rmse'][s][1].values[0]
                        small_rmse[1] = small_rmse[1] + 1
                        
                    if s in cfg.mid_sno:
                        #print('medium')
                        mid_rmse[0] = mid_rmse[0] + r['pred_rmse'][s][1].values[0]
                        mid_rmse[1] = mid_rmse[1] + 1
                        
                    if s in cfg.big_sno:
                        #print('big')
                        big_rmse[0] = big_rmse[0] + r['pred_rmse'][s][1].values[0]
                        big_rmse[1] = big_rmse[1] + 1
                final_time = final_time + r['train_time']
        
        
        print('model: ', title_list[i])
        print('small:', small_rmse[0]/small_rmse[1])
        print('mid:', mid_rmse[0]/mid_rmse[1])
        print('big:', big_rmse[0]/big_rmse[1])
        print('avg:', (big_rmse[0]+mid_rmse[0]+small_rmse[0])/(big_rmse[1]+mid_rmse[1]+small_rmse[1]))
        print("time:", final_time)
        print("-----------------------------------------------")
        
#%%
if False:
    with open('models_onestastiononemodel_allstations_all_feature.pickle', 'wb') as handle:
        pickle.dump(all_res_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
import pickle
if False:
    with open('pickle/rf_onestastiononecluster_allfeature_result.pickle', 'rb') as f:
        all_res_lst = pickle.load(f)

#%%
    lasso_time = 0
    ridge_time = 0
    rf_time = 0
    for i, estimator in enumerate(estimator_list):
        for r in all_res_lst:
            if title_list[i] == r['model']['name']:
                if title_list[i] == 'Lasso ':
                    lasso_time = lasso_time + r['train_time']
                if title_list[i] == 'Ridge ':
                    ridge_time = ridge_time + r['train_time']
                if title_list[i] == 'Random Forest ':
                    rf_time = rf_time + r['train_time']
                    
    print("Lasso time:", lasso_time)
    print("Ridge time:", ridge_time)
    print("RF time:", rf_time)

#%%
if False:
    #lasso_coef_df.to_csv("lasso_coef2.csv")
    mean_dict = {}
    for col in lasso_coef_df.columns:
        mean_dict[col] = lasso_coef_df[col].mean()
    
    mean_df = pd.Series(mean_dict).sort_values()

    #print(mean_df.describe())
    for c in [0.2, 0.5]:
        idx = mean_df[abs(mean_df) > c].index
        print('c:',c)
        print('n of feature:',len(idx))
        print(idx)
    #mean_df.plot(kind='bar',figsize=(60,6))

#print(features)
#print(len(features))

#%%
if False:
    rmse_df = pd.read_csv("data/feature_rmse_result.csv")
    
    lasso_small_df = rmse_df[(rmse_df['station scale']=='small') &(rmse_df['model']=='lasso')]
    lasso_small_df.plot(kind='bar')
    
    small_df = rmse_df[(rmse_df['station scale']=='small')]
    small_df = small_df.set_index(small_df['model'])
    small_df.plot(kind='bar',figsize=(22,7))
    
    mid_df = rmse_df[(rmse_df['station scale']=='mid')]
    mid_df = mid_df.set_index(mid_df['model'])
    mid_df.plot(kind='bar',figsize=(22,7))
    
    large_df = rmse_df[(rmse_df['station scale']=='large')]
    large_df = large_df.set_index(small_df['model'])
    large_df.plot(kind='bar',figsize=(22,7))