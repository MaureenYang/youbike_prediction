import json,time,math,sys,os
import numpy as np
import pandas as pd
import datetime as dt
import sqlite3

import ub_config as cfg
from backend import backend
from youbike_processor import YoubikeProcessor
from weather_processor import WeatherProcessor
from weather_web_processor import WeatherWebProcessor


'''weather data from json to csv'''
def weather_station_rt_2csv(): 
    pass
    

'''weather data from web to csv'''
def weather_station_web_2csv(): 
    wprocessor = WeatherWebProcessor()
    source_folder = cfg.raw_weather_web_path
    target_folder = cfg.csv_weather_web_path

    for wsid in cfg.weather_station_dict:
        print(wsid)
        weather_df = pd.DataFrame()
        filesinpath = os.listdir(source_folder)
        for file in filesinpath:
            print(file)
            if file.startswith('web_'+wsid):
                wprocessor.read(source_folder+file)
                w_data = wprocessor.get_dict()
                weather_df = weather_df.append(w_data,ignore_index=True)

        weather_df.to_csv(target_folder +"weather_web_"+wsid+".csv")

''' ubike data from raw to csv'''
def ubike_from_raw_2csv():
    pass



''' ubike data from db to csv'''
def ubike_from_db_2csv():
    
    con = sqlite3.connect('db/youbike-20210611.db')
    cur = con.cursor()
       
    for stationid in range(1,405):
        print('station ', str(stationid),' start...')
        ubike_df = pd.DataFrame()
        #for row in cur.execute('SELECT * FROM observations ORDER BY station_id'):
        for row in cur.execute('SELECT * FROM observations WHERE station_id ==' + str(stationid)):
            epoch_time = row[1]
            time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_time))
            print("Formatted Date:", time_formatted)
            r_dict = {'station_id':row[0],'time':time_formatted,'sbi':row[2]}
            ubike_df = ubike_df.append(r_dict, ignore_index=True)
                
        ubike_df.to_csv(cfg.csv_ubike_db_path + 'ubike_db_sno_'+str(stationid).zfill(3)+'.csv')
        
        
       
def ubike_weather_merge():
    be = backend()
    be.update_all_station()
    station_list = be.get_all_station_list() #get staion combination
    mapping_table = {}
    station_sno_list = []
    
    for key,val in station_list.items():
       try:
           #print(key, val)
           station_sno_list = station_sno_list + [key]
           ws, ws2 = val.get_weather_ids()
           mapping_table[key] = (ws, ws2)
           
       except Exception as e:
           print(e)
           
    #print(mapping_table)
    
    for stationid in range(1,405):#cfg.station_sno_list:
        try:
            ubike_df = pd.read_csv(cfg.csv_ubike_db_path+'ubike_sno_'+str(stationid).zfill(3)+'.csv')
            
            # resample
            print('start to resampe')
            ubike_df['time'] = pd.to_datetime(ubike_df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
            ubike_df = ubike_df.set_index(pd.DatetimeIndex(ubike_df['time'])).drop(columns=['time','Unnamed: 0'])
            ubike_df= ubike_df['20200714':'20210611']
            grouped_ubike_df = ubike_df.resample('60min').mean().interpolate().round(0).astype(int)
            #print(grouped_ubike_df.head())
            
            # according to mapping table and open file
            wdf = pd.read_csv(cfg.csv_weather_web_path+"weather_web_"+mapping_table[stationid][1]+".csv")
            wdf2 = pd.read_csv(cfg.csv_weather_web_path+"weather_web_"+mapping_table[stationid][0]+".csv")
            print(mapping_table[stationid][1])
            print(mapping_table[stationid][0]) 
            wdata1 = wdf.drop(columns=['Unnamed: 0']).set_index('time')
            wdata2 = wdf2.drop(columns=['Unnamed: 0']).set_index('time')
            #print(wdata2)                  
            
            #delete unnecessary field
            wdata2_short = wdata2.drop(columns=cfg.weather_feature_s1)
            wdata1_short = wdata1.drop(columns=cfg.weather_feature_s2 +　['location'])
        
            weather_group = pd.merge(wdata1_short,wdata2_short,left_index=True, right_index=True)
            weather_group = weather_group.drop(columns=['locationName_x','locationName_y','location','stationId_x','stationId_y'])
            weather_group.index = pd.to_datetime(weather_group.index, format='%Y-%m-%d %H:%M:%S', errors='ignore')
            weather_group = weather_group.sort_index()
            group_data =pd.merge(grouped_ubike_df,weather_group,left_index=True, right_index=True)
            #print(group_data.head())
            
            filename = 'merged_sno_'+str(stationid).zfill(3)+'_data.csv'
            group_data.to_csv(cfg.csv_merged_db_web_path + filename)
            print(filename + " saved!")
            
        except Exception as e:
            print('error[',str(stationid),']:',e)



if __name__ == "__main__":
    
    ''' 
        this file is for data preprocessing
        1. parsing ubike data to csv
            1)json 
            2)database: done
        2. parsing weather data from 
            1)opendata json format
            2)website information: done
        3. merge the youbike data and weather data to csv file: done
            
    '''
    #weather_station_web_2csv()
    #ubike_weather_merge()
    #ubike_from_db_2csv()
    
    pass
