# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:47:17 2021

@author: Maureen
"""

db_root_path = "E:/ubike_pred_db/"
csv_file_path = db_root_path + "csv_data/"
raw_file_path = db_root_path + "raw_data/"

raw_weather_web_path = raw_file_path + "weather_web_2021/"

csv_weather_web_path = csv_file_path + "weather_web_2021/"
csv_ubike_db_path = csv_file_path + "ubike_db_2021/"
csv_merged_db_web_path = csv_file_path + "merged_db_web_2021/"
csv_parsed_db_web_path = csv_file_path + "parsed_ubike_db_web_2021/"

code_path = "D:/git/youbike_prediction/"
result_path = code_path + "result/"
result_plot_path = result_path+"plot/"
result_plot_pred_path = result_plot_path + "predict_result/"

ipython_path = code_path+"ipython/csv/"



weather_feature_s1 = ['CloudA','GloblRad','PrecpHour','SeaPres','UVI','Visb','WDGust','WSGust','td']
weather_feature_s2 = ['HUMD','H_24R','PRES','TEMP','WDIR','WDSE']
weather_feature_all = weather_feature_s1 + weather_feature_s2
weather_feature_used = ['PrecpHour','UVI','Visb','td', 'HUMD','H_24R','PRES','TEMP','comfort','WDSE']

weather_station_dict = {
    
    "466910" : {'id':"466910", 'name':'鞍部', 'coor': [121.5297,25.1825]},
    "466920" : {'id':"466920", 'name':'臺北', 'coor': [121.5148,25.0376]},
    "466930" : {'id':"466930", 'name':'竹子湖', 'coor': [121.5445,25.1620]},    
  
    "C0A980" : {'id':"C0A980", 'name':'社子', 'coor': [121.4696,25.1095]},
    "C0A9E0" : {'id':"C0A9E0", 'name':'士林', 'coor': [121.5030,25.0903]},
    "C0A9F0" : {'id':"C0A9F0", 'name':'內湖', 'coor': [121.5755,25.0794]},   
    
    "C0AH40" : {'id':"C0AH40", 'name':'平等', 'coor': [121.5771,25.1291]},
    "C0AH70" : {'id':"C0AH70", 'name':'松山', 'coor': [121.5504,25.0487]},
    "C0AI40" : {'id':"C0AI40", 'name':'石牌', 'coor': [121.5132,25.1156]},   
    
    "C0AC40" : {'id':"C0AC40", 'name':'大屯山', 'coor': [121.5224,25.1757]},
    "C0AC70" : {'id':"C0AC70", 'name':'信義', 'coor': [121.5645,25.0378]},
    "C0AC80" : {'id':"C0AC80", 'name':'文山', 'coor': [121.5757,25.0023]},   
    "C0A9C0" : {'id':"C0A9C0", 'name':'天母', 'coor': [121.5371,25.1174]},   
}

weather_station_list = [
    
    {'id':"466910", 'name':'鞍部', 'coor': [121.5297,25.1825]},
    {'id':"466920", 'name':'臺北', 'coor': [121.5148,25.0376]},
    {'id':"466930", 'name':'竹子湖', 'coor': [121.5445,25.1620]},    
  
    {'id':"C0A980", 'name':'社子', 'coor': [121.4696,25.1095]},
    {'id':"C0A9E0", 'name':'士林', 'coor': [121.5030,25.0903]},
    {'id':"C0A9F0", 'name':'內湖', 'coor': [121.5755,25.0794]},   
    
    {'id':"C0AH40", 'name':'平等', 'coor': [121.5771,25.1291]},
    {'id':"C0AH70", 'name':'松山', 'coor': [121.5504,25.0487]},
    {'id':"C0AI40", 'name':'石牌', 'coor': [121.5132,25.1156]},   
    
    {'id':"C0AC40", 'name':'大屯山', 'coor': [121.5224,25.1757]},
    {'id':"C0AC70", 'name':'信義', 'coor': [121.5645,25.0378]},
    {'id':"C0AC80", 'name':'文山', 'coor': [121.5757,25.0023]},   
    {'id':"C0A9C0", 'name':'天母', 'coor': [121.5371,25.1174]},   
]


ignore_list = [15, 20, 160, 198, 199, 200] # no station
ignore_list2 = [28, 47, 58, 69, 99, 101 ,106 ,153 , 168 ,185, 190, 239, 240,264,306,311, 313,346,378,382,383,387]
station_sno_list = set(range(1,405)) - set(ignore_list) - set(ignore_list2)

