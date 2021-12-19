# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:52:13 2021

@author: Maureen
"""

import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import ub_config as cfg

filepath = cfg.arima_result_path
station_list = range(1,405)#cfg.station_sno_list
#station_list = cfg.small_station_sno_list

# Read all predict result and compare

def cal_rmse(ans, pred):
    #print(ans)
    #print(pred)
    rmse = np.sqrt(mean_squared_error(ans, pred))
    return rmse

#%%

res_df = pd.DataFrame()

for sno in station_list:
    try:
        f = 'sarima_prediction_'+str(sno).zfill(3)+'.csv'
        print("filename:", f) 
        y_df = pd.read_csv(filepath + f)
        y_df['time'] = pd.to_datetime(y_df['Unnamed: 0'], format='%Y/%m/%d %H%M%S', errors='ignore')
        y_df = y_df.set_index(pd.DatetimeIndex(y_df['time'])).drop(columns=['Unnamed: 0','time'])
        
        if y_df[y_df > 200].any()[0]:
            print("filename:", f) 
            print(y_df['0'][y_df['0'] > 200])
        
        
        res_df['y_sbi_' + str(sno).zfill(3)] = y_df['0']
        
        #print(res_df)        

    except Exception as e:
        print(f)

res_df.to_csv("result_sarima.csv")

#%%
#generate answer:
ans_f = ['result_answer.csv']
files1 = ['result/result_lasso.csv' , 'result/result_ridge.csv', 'result/result_rf.csv']
files2 = ['result_sarima.csv']

#for each model

#%%
ans_df = pd.read_csv(ans_f[0])
ans_df['time'] = pd.to_datetime(ans_df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
ans_df = ans_df.set_index(pd.DatetimeIndex(ans_df['time'])).drop(columns=['time'])
print(ans_df)

#%%
f_df = pd.DataFrame()
for f in files1:
    rmse_dict = {}
    pred_df = pd.read_csv(f)
    print(f)
    print(pred_df)
    pred_df['time'] = pd.to_datetime(pred_df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['time'])).drop(columns=['time'])

    for col in pred_df.columns:
        print('col:',col)
        pidx = pred_df[col].index[pred_df[col].apply(np.isnan)]
        aidx = ans_df[col].index[ans_df[col].apply(np.isnan)]
        
        if len(pidx) != 0 or len(aidx) != 0:
            #print('col:',col)
            #print(pidx)
            continue
        rmse  = cal_rmse(ans_df[col],pred_df[col])
        rmse_dict[col] = rmse
        print('rmse:',rmse)

    f_df[f] = pd.Series(rmse_dict)



for f in files2:
    rmse_dict = {}
    pred_df = pd.read_csv(f)
    print(f)
    print(pred_df)
    pred_df['time'] = pd.to_datetime(pred_df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['time'])).drop(columns=['time'])

    for col in pred_df.columns:   
        pidx = pred_df[col].index[pred_df[col].apply(np.isnan)]
        aidx = ans_df[col].index[ans_df[col].apply(np.isnan)]
        print('col:',col)
        if len(pidx) != 0 or len(aidx) != 0:
            #print('col:',col)
            #print(pidx)
            continue
        rmse  = cal_rmse(ans_df[col],pred_df[col][:-1])
        rmse_dict[col] = rmse
        print('rmse:',rmse)

    f_df[f] = pd.Series(rmse_dict)
        
    
#f_df.to_csv('models_rmse.csv')

#%%

f_df = f_df.dropna()
#%%
f3_df_dict = {}
for f in files1:
    f3_df_dict[f] = pd.read_csv(f)
    
for sno in station_list:
    try:
        col_name = 'y_sbi_'+str(sno).zfill(3)
        col_df = pd.DataFrame()
        for f in files1:
            pred_df = f3_df_dict[f]
            pred_df['time'] = pd.to_datetime(pred_df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
            pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['time'])).drop(columns=['time'])
            
            col_df[f] = pred_df[col_name]
            
            col_df.to_csv(cfg.model_result_path +"pred_result_sno_"+str(sno).zfill(3)+".csv")
    except Exception as e:
        print(e)

#%%

rmse_dict = {}
for sno in station_list:
    try:
        r_df = pd.read_csv(cfg.model_result_path +"pred_result_sno_"+str(sno).zfill(3)+".csv")
        
        r_df['avg'] = r_df.mean(axis=1)
        rmse  = cal_rmse(ans_df['y_sbi_'+str(sno).zfill(3)],r_df['avg'])
        rmse_dict['y_sbi_'+str(sno).zfill(3)] = rmse
    except Exception as e:
        print(e)
    
f_df['avg'] = pd.Series(rmse_dict)
f_df.to_csv('models_rmse.csv')  
    
    
#%%
#get tot
tot_f = pd.read_csv("D:/git/youbike_prediction/result/all_result.csv")
tot_f = tot_f[['sno','tot']]
tot_f = tot_f.set_index(tot_f['sno']).drop(columns=['sno'])

#%%
print(len(f_df))
print(len(tot_f))
#%%
pert_df = f_df
#%%
for row in tot_f.iterrows():
    try:
        tot = row[1].values[0]
        sno = int(row[0])
        #print(pert_df.loc['y_sbi_001',])
        	
        pert_df.loc['y_sbi_' + str(sno).zfill(3),] = pert_df.loc['y_sbi_' + str(sno).zfill(3),]/tot
        #break
    except Exception as e:
        print(e)
        
#%%
print(pert_df)    
pert_df.to_csv("model_rmse_percentage.csv")

#%%
pert_df = pert_df.dropna()
if True:
   for m in pert_df.columns:
        small_rmse = [0,0]
        mid_rmse = [0,0]
        big_rmse = [0,0]
        for r, val in pert_df[m].iteritems():
            #print(r, val)
            s = [int(a) for a in r.split('_') if a.isdigit()]
            s = s[0]
            #print(s)
            if s in cfg.small_sno:
                #print('small')
                small_rmse[0] = small_rmse[0] + val
                small_rmse[1] = small_rmse[1] + 1
                        
            if s in cfg.mid_sno:
                #print('medium')
                mid_rmse[0] = mid_rmse[0] + val
                mid_rmse[1] = mid_rmse[1] + 1
                        
            if s in cfg.big_sno:
                #print('big')
                big_rmse[0] = big_rmse[0] + val
                big_rmse[1] = big_rmse[1] + 1
        
        
        print('model: ', m)
        print('small:', small_rmse[0]/small_rmse[1])
        print('mid:', mid_rmse[0]/mid_rmse[1])
        print('big:', big_rmse[0]/big_rmse[1])
        print('avg:', (big_rmse[0]+mid_rmse[0]+small_rmse[0])/(big_rmse[1]+mid_rmse[1]+small_rmse[1]))

        print("-----------------------------------------------")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    