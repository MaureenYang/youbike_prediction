# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:09:25 2021

@author: Maureen
"""
import seaborn as sns
import pandas as pd
from utility import data_preprocess

ignore_list = [15, 20, 160, 198, 199, 200] # no station
ignore_list2 = [28, 47, 58, 69, 99, 101 ,106 ,153 , 168 ,185 ,190 ,229 ,239,240 ,264 ,306 ,311 ,313,346,378,382,383,387]
station_sno_list = set(range(1,405)) - set(ignore_list) - set(ignore_list2)
filepath = "E:/ubike_pred_db/csv_data/"
train_start_date = '20200521 13:00:00'
train_end_date = '20210131 23:00:00'

data = pd.DataFrame()
for i in station_sno_list: 
    if True:
        print('i:', i)
        ndf = pd.read_csv(filepath + "parsed_normalized_ubike_db_web_2021/parsed_sno_"+str(i).zfill(3)+".csv")
        ndf['time'] = pd.to_datetime(ndf['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        ndf = ndf.set_index(pd.DatetimeIndex(ndf['time'])).drop(columns=['time'])
        ndf = ndf[train_start_date:train_end_date]
        data = data.append(ndf)
    else:
        f = 'merged_db_web_2021/merged_sno_'+str(i).zfill(3)+'_data.csv'
        print("filename:", f)
        df = pd.read_csv(filepath + f)
        df = data_preprocess(df,normalize=True)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])
        df.to_csv(filepath + "parsed_normalized_ubike_db_web_2021/parsed_sno_"+str(i).zfill(3)+".csv")
        #break
    
    
#%%
print(data.columns)
#%%
from lasso_model import lasso

final_drop_list =  ['cat_hospital_500', 'cat_college_500',
       'cat_train_500', 'cat_hospital_1000', 'cat_college_1000', 'cat_train_1000','tot','station_id'] 
dd = data.drop(columns = final_drop_list)
train_x = dd.dropna()
train_x = train_x.drop(columns = ['y_sbi'])
train_y = dd[['y_sbi']]

train_x_wo_t = train_x.reset_index().drop(columns=['time'])
train_y_wo_t = train_y.reset_index().drop(columns=['time'])

#%%
lasso_model, results, best_param, indexing = lasso(train_x_wo_t, train_y_wo_t,rfecv_en=False)
   
label_name = train_x_wo_t.columns  
#%%
coef_lab = pd.Series(lasso_model.coef_,index=label_name)
#%%

coef_lab.to_csv("lasso_coef.csv")
#%%
print(coef_lab)
#coef_lab = coef_lab.T
#coef_lab.plot(kind='bar',fig_size=(20,7))
sorted_coef = coef_lab.sort_values()
print(sorted_coef)
#%%
#ax = sorted_coef.plot.bar(rot=1, figsize=(20,7))


sns.set(rc={"figure.figsize": (20, 6)})
ax = sns.barplot(sorted_coef.index, sorted_coef.values, palette="Blues")

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



