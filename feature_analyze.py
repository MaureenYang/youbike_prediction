# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 09:02:30 2021

@author: Maureen
"""
import pandas as pd
import ub_config as cfg
from utility import ubike_read_from_file
import os, sys
import numpy as np
import matplotlib.pyplot as plt


station_list = cfg.station_sno_list
feature_tag = ['PrecpHour','H_24R','td','HUMD','comfort','UVI','PRES','WDSE','TEMP'] 


def Average(lst):
    if len(lst) == 0:
        return 0
    return (sum(lst) / len(lst))


def getdtsetbytag(df, tag):
    
    if tag == 'H_24R':
        dt_list = ((set(df[(df[tag] > 0)].index)))
    
    if tag == 'PrecpHour':
        dt_list = ((set(df[(df[tag] > 0)].index)))
        
    if tag == 'td':
        dt_list = ((set(df[(df[tag] > 20)].index)))
    
    if tag == 'HUMD':
        dt_list = ((set(df[(df[tag] > 76)].index)))
        
    if tag == 'comfort':
        dt_list = ((set(df[(df[tag] != 3)].index)))
    
    if tag == 'UVI':
        dt_list = ((set(df[(df[tag] > 4)].index)))

    if tag == 'PRES':
        dt_list = ((set(df[(df[tag] < 1008)].index)))
    
    if tag == 'WDSE':
        dt_list = ((set(df[(df[tag] >= 8)].index)))
        
    if tag == 'TEMP':
        dt_list = ((set(df[(df[tag] >= 29) | (df[tag] < 20)].index)))
        

    return dt_list

def generate_describe_csv():
    
    df_dict = {}
    for tag in feature_tag:
        df = pd.DataFrame()
        df_dict[tag] = df
    for sno in station_list:
        try:
            df = ubike_read_from_file(sno)
            for tag in feature_tag:
                df_dict[tag][sno] = df[tag].describe()
        except Exception as e:
            print('[error]',e)
            print('station no:',sno)
            #pass

    for tag in feature_tag:
        df_dict[tag].T.to_csv("describe_"+tag+".csv")


'''todo'''
def draw_specific_condition_by_hour(df, tag):

    a = df[(df[tag] == 3)].index.hour.values
    unique, counts = np.unique(a, return_counts=True)
    maps_dict = dict(zip(unique, counts))
    keys = maps_dict.keys()
    values = maps_dict.values()    
    plt.bar(keys, values)
    

'''
for each station, each feature: [avg, len of avg ,avg_no, len of no avg]

'''
def get_all_feature_station_df():
    
    res_df = pd.DataFrame()
    for sno in station_list:
        try:
            ser_list = []
            df = ubike_read_from_file(sno)
            for tag in feature_tag:
                print('tag:', tag)
                new_df = df[[tag,'percet']]
                
                h24r_dt_set = getdtsetbytag(df,tag)
                h24r_dt_list = list(sorted(h24r_dt_set))
                
                des_h = {}
                for d in h24r_dt_list:
                    des_h[d.strftime("%Y%m%d")] = new_df.loc[d.strftime("%Y%m%d")].percet.mean()
        
                
                normal_dt_list = list(set(new_df.index)- h24r_dt_set)            
                des_n = {}
                for d in normal_dt_list:
                    des_n[d.strftime("%Y%m%d")] = new_df.loc[d.strftime("%Y%m%d")].percet.mean()


                if len(list(des_h.values())) == 0:
                    hval = 0
                else:
                    hval = Average(list(des_h.values()))
                    
                if len(list(des_n.values())) == 0:
                    nval = 0
                else:
                    nval = Average(list(des_n.values()))

                res = [hval,len(list(des_h.values())), nval, len(list(des_n.values()))]
                ser_list = ser_list + [res]
                print(ser_list)
            x = pd.Series(ser_list, index=feature_tag)
        
            res_df = res_df.append(x,ignore_index=True)
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno) 
    
    res_df['sno'] = station_list
    res_df = res_df.set_index('sno')
    
    return res_df


def compare_each_feature_for_stations(df, tag ,plotbysno=True,plotbypercent=True,savecsv=True):
    

    res_df = pd.DataFrame()
    res_df['avg']= df[tag].apply(lambda x: x[0])
    res_df['avg_no']= df[tag].apply(lambda x: x[2])
    res_df['percentage']= df[tag].apply(lambda x: (x[0]-x[2])/x[0])

        
    if plotbysno:
        fig = res_df.drop(['percentage'],axis=1).plot.bar(figsize=(len(res_df.index),7),rot=0).get_figure()
        fig.savefig('feature_plot_by_sno_'+tag+'.png')
            
    if savecsv:
        res_df.to_csv("feature_analyze_"+tag+".csv")
            
    if plotbypercent:
        res_df_p = res_df.drop(['avg','avg_no'],axis=1).sort_values(by='percentage')
        fig2 = res_df_p.plot.bar(figsize=(round(len(res_df.index)/2),7),rot=0).get_figure()
        fig2.savefig('feature_plot_by_percentage_'+tag+'.png')
                

def compare_allfeature_for_allstations(df,savecsv=False):
    try:
        
        new_s = pd.DataFrame()
        for col in df.columns:

            res_df = pd.Series()
            res_df['avg_p'] = df[col].apply(lambda x: x[0]).mean()
            res_df['avg_no_p'] = df[col].apply(lambda x: x[2]).mean()
            new_s = new_s.append(res_df, ignore_index=True)
            
        name_map = {}
        for i in range(0,len(new_s.columns)):
            name_map[i] = feature_tag[i]
        print(name_map)
        new_s.rename(index= name_map, inplace=True)
        ax = new_s.plot.bar(rot=0,figsize=(len(new_s.index),7))
        if savecsv:
            new_s.to_csv("feature_avg_sum_result.csv")   
                
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



'''
   non-weather features
   - find 
   1. holiday: true/false
   2. wkdy_0: monday
   
'''

# distrubution

def temp_function(df,tag):
    
    #get the unique value of the tag
    for val in df[tag].unique():
        a = df[(df[tag] == val)]
        keys = []
        values = []
        for i in range(0,24):
            b = round(a[a.index.hour == i].percet.mean())

            keys = keys + [i]
            values = values + [b]
            
        fig = plt.bar(keys, values)
        plt.bar(keys, values,color=['blue'])
        plt.xlabel('hour')
        plt.ylabel('percentage(%)')
        plt.title('Distrubution of ' + tag +'('+str(val)+') in station('+str(df.station_id[0])+')')
        #plt.show()
        plt.savefig(cfg.result_path + 'plot/'+tag+'/distrubution_' + tag +'_'+str(val)+'_station'+str(df.station_id[0])+'.png')

wkdy=['wkdy_0','wkdy_1','wkdy_2','wkdy_3','wkdy_4','wkdy_5','wkdy_6']

def temp2_function(df,tag_list):
    
    #get the unique value of the tag
    for tag in tag_list:
        a = df[(df[tag] == 1)]
        keys = []
        values = []
        for i in range(0,24):
            b = round(a[a.index.hour == i].percet.mean())

            keys = keys + [i]
            values = values + [b]
        fig = plt.bar(keys, values)
        plt.bar(keys, values,color=['blue'])
        plt.xlabel('hour')
        plt.ylabel('percentage(%)')
        plt.title('Distrubution of ' + tag + ' in station('+str(df.station_id[0])+')')
        #plt.show()
        plt.savefig(cfg.result_path + 'plot/'+tag+'/distrubution_' + tag +'_'+'_station'+str(df.station_id[0])+'.png')
        

if __name__ == '__main__':
    
    # create describe csv for the features
    # observe the feature by describe
    # create plot/csv for each station, sort by value/station
    # create plot/csv for all station average 
    #dd = get_all_feature_station_df()
    #compare_each_feature_for_stations(dd,'H_24R',plotbysno=True,plotbypercent=True,savecsv=True)   
    #compare_allfeature_for_allstations(dd,savecsv=False)
    import os,sys
    for tag in wkdy:
        try:
            # Create target Directory
            os.mkdir(cfg.result_path + 'plot/'+tag+'/')
            print("Directory " , tag ,  " Created ") 
        except FileExistsError:
            print("Directory " , tag ,  " already exists")

    for tag in ['holiday']:
        try:
            # Create target Directory
            os.mkdir(cfg.result_path + 'plot/'+tag+'/')
            print("Directory " , tag ,  " Created ") 
        except FileExistsError:
            print("Directory " , tag ,  " already exists")
            
    for sno in station_list:
        df = ubike_read_from_file(sno)
        temp_function(df,"holiday")
        temp2_function(df,wkdy)
        
    pass



