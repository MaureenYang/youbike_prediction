#
#   filename: weather_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#

import os
import json
import xml.etree.ElementTree
import datetime
from base_processor import base_processor


def getkeyvalue(src):
    new_element = {}
    key_name = src['elementName']
    try:
        new_element[key_name] = float(src['elementValue']['value'])
    except:
        new_element[key_name] = src['elementValue']['value']
    return new_element

class WeatherProcessor(base_processor):
    """
        this is description of Weather processor
        input:
         - filename
         - old: old format/new format

    """
    __type__ = 'Weather Processor'

    stationIdList = ["466910","466920","466930","C0A980","C0A9E0","C0A9F0","C0AH40","C0AH70","C0AI40", "C0AC40", "C0AC70", "C0A9C0"]
    weatherPara = ['PRES','TEMP','HUMD','WDSD','WDIR','H_FX','H_XD','H_24R','24R','D_TS','VIS','H_UVI']
    weatherPara_map = {'PRES':'PRES','TEMP':'TEMP','HUMD':'HUMD','WDSD':'WDSE','WDIR':'WDIR','H_FX':'WSGust','H_XD':'WDGust','H_24R':'H_24R','24R':'24R','D_TS':'PrecpHour','VIS':'Visb','H_UVI':'UVI'}

    def __init__(self):
        super().__init__()

    def read(self, filename, old=False):
        try:
            json_file = open(filename, encoding='utf-8')
            dict_data = json.load(json_file)
            json_file.close()
        except Exception as e:
            print(e)

        if old:
            dict_tmp = {}
            try:
                dict_tmp['stationId'] = dict_data['stationId']
                dict_tmp['locationName'] = dict_data['locationName']
                coor = self.get_station_coor(dict_data['stationId'])
                dict_tmp['lat'] = coor[1]
                dict_tmp['lon'] = coor[0]
                dict_tmp['time'] = datetime.datetime.strptime(dict_data['time'], "%Y-%m-%dT%H:%M:%S")
                dict_tmp['PRES'] = self.tran2float(dict_data['PRES'])
                dict_tmp['TEMP'] = self.tran2float(dict_data['TEMP'])
                dict_tmp['HUMD'] = self.tran2float(dict_data['HUMD'])
                dict_tmp['WDSE'] = self.tran2float(dict_data['WDSE'])
                dict_tmp['WDIR'] = self.tran2float(dict_data['WDIR'])
                dict_tmp['H_24R'] = self.tran2float(dict_data['H_24R'])

                dict_tmp['WSGust'] = self.tran2float(dict_data['WSGust'])
                dict_tmp['WDGust'] = self.tran2float(dict_data['WDGust'])
                dict_tmp['PrecpHour'] = self.tran2float(dict_data['PrecpHour'])
                dict_tmp['Visb'] = self.tran2float(dict_data['Visb'])
                dict_tmp['UVI'] = self.tran2float(dict_data['UVI'])

            except Exception as e:
                print(e)

            self.__data_dict__ = dict_tmp

        else:
            dict_tmp = []
            try:
                for key,value in dict_data['cwbopendata'].items():
                    try:
                        if(key == 'sent'):
                            time = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S+08:00")
                        if(key =='location'):
                            taipei_list = self.find_taipei(value)
                            dict_tmp = self.station_parsing(taipei_list)

                    except Exception as e:
                        print(e)
                        print('error: ',value)
                        dict_tmp = None

                self.__data_dict__ = dict_tmp

            except os.error as e:
                self.__data_dict__ = None

    def get_dict(self):
        return self.__data_dict__

    def get_station_coor(self, stationId):
        
        if stationId == "466910":#ok
            return [121.5297, 25.1825]
        
        if stationId == "466920":#ok
            return [121.5148, 25.0376]
        if stationId == "466930":#ok
            return [121.5445, 25.1620]

        if stationId == "C0A980":#ok
            return [121.4697, 25.1095]
        
        #if stationId == "C0A9A0":
        #    return [121.5429, 25.0780]
        #if stationId == "C0A9B0":
        #    return [121.5138, 25.1163]
        
        if stationId == "C0A9C0":#ok
            return [121.5372, 25.1175]
                
        if stationId == "C0A9E0": #ok
            return [121.5030, 25.0903]
        if stationId == "C0A9F0": #ok
            return [121.5755, 25.0794]

        if stationId == "C0AC40":#ok
            return [121.5224, 25.1757]
        if stationId == "C0AC70":#ok
            return [121.5757, 25.0024]
        if stationId == "C0AC80":#ok
            return [121.5646, 25.0378]

        if stationId == "C0AH40":#ok
            return [121.5771, 25.1291]
        if stationId == "C0AH70":#ok
            return [121.5504, 25.0487]
        if stationId == "C0AI40":#ok
            return [121.5132, 25.1156]
        
        #if stationId == "C1A730":
        #    return [121.5395, 25.0143]
        #if stationId == "C1AC50":
        #    return [121.4693, 25.1334]

        return None

    def station_parsing(self,list): # for new format
        stationlist = []
        for data in list:
            newdata = {}
            newdata['stationId'] = data['stationId']
            newdata['locationName'] = data['locationName']
            newdata['lat'] = float(data['lat'])
            newdata['lon'] = float(data['lon'])
            newdata['time'] = datetime.datetime.strptime(data['time']['obsTime'], "%Y-%m-%dT%H:%M:%S+08:00")
            wdata = self.weather_element(data['weatherElement'])
            newdata.update(wdata)
            stationlist = stationlist + [newdata]

        return stationlist

    def weather_element(self,list):
        ele_dict = {}
        for data in list:
            if data['elementName'] in self.weatherPara:
                ele_dict[self.weatherPara_map[data['elementName']]] = getkeyvalue(data)[data['elementName']]

        return ele_dict

    def find_taipei(self,list):
        tmplist = []
        for data in list:
            if data['stationId'] in self.stationIdList:
                tmplist = tmplist + [data]

        return tmplist

    def find_taipei_id(self,list):
        for data in list:
            foundflag = False
            pdata = data['parameter']
            for ppdata in pdata:
                if ppdata['parameterName'] == "CITY" and ppdata['parameterValue'] == "臺北市":
                    foundflag = True
            if foundflag:
                print(data['stationId'])

    def tran2float(self, str_val):
        if str_val is None:
            return None
        try:
            val = float(str_val)
            return val

        except ValueError:
            return None