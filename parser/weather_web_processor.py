#
#
#   filename: weather2_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#
import os
import json
import datetime
from base_processor import base_processor


class WeatherWebProcessor(base_processor):
    """
        this is description of Weather processor from website
    """
    __type__ = 'WeatherWebProcessor'

    def __init__(self):
        super().__init__()

    def tran2float(self,str_val):
        if str_val is None:
            return None
        try:
            val = float(str_val)
            return val
        except ValueError:
            return None

    def read(self,filename):
        try:
            f = open(filename, 'r')
            jdata = f.read()
            f.close()
            try:
                dict_data = json.loads(jdata)
            except:
                self.__data_dict__ = None
                return

            try:
                dict_data['time'] = datetime.datetime.strptime(dict_data['time'], "%Y-%m-%dT%H:%M:%S")
                dict_data['SeaPres'] = self.tran2float(dict_data['SeaPres'])
                dict_data['td'] = self.tran2float(dict_data['td'])
                dict_data['PRES'] = self.tran2float(dict_data['PRES'])
                dict_data['TEMP'] = self.tran2float(dict_data['TEMP'])
                dict_data['HUMD'] = self.tran2float(dict_data['HUMD'])
                dict_data['WDSE'] = self.tran2float(dict_data['WDSE'])
                dict_data['WDIR'] = self.tran2float(dict_data['WDIR'])
                dict_data['H_24R'] = self.tran2float(dict_data['H_24R'])
                dict_data['WSGust'] = self.tran2float(dict_data['WSGust'])
                dict_data['PrecpHour'] = self.tran2float(dict_data['PrecpHour'])
                dict_data['GloblRad'] = self.tran2float(dict_data['GloblRad'])
                dict_data['Visb'] = self.tran2float(dict_data['Visb'])
                dict_data['UVI'] = self.tran2float(dict_data['UVI'])
                dict_data['CloudA'] = self.tran2float(dict_data['CloudA'])

                if dict_data['stationId'] == "466910":
                    dict_data['location'] = {'type':'Point','coordinates':[121.5297,25.1825]}
                if dict_data['stationId'] == "466920":
                    dict_data['location'] = {'type':'Point','coordinates':[121.5148,25.0376]}
                if dict_data['stationId'] == "466930":
                    dict_data['location'] = {'type':'Point','coordinates':[121.5445,25.1620]}
                    
                if dict_data['stationId'] == "C0A980":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.4697,25.1095]}
                if dict_data['stationId'] == "C0A9A0":
                    dict_data['location'] = {'type':'Point','coordinates':[121.5429,25.0780]}
                if dict_data['stationId'] == "C0A9B0":
                    dict_data['location'] = {'type':'Point','coordinates':[121.5138,25.1163]}
                if dict_data['stationId'] == "C0A9C0":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5372,25.1175]}
                if dict_data['stationId'] == "C0A9E0":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5030,25.0903]}
                if dict_data['stationId'] == "C0A9F0":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5755,25.0794]}


                if dict_data['stationId'] == "C0AC40":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5224,25.1757]}
                if dict_data['stationId'] == "C0AC70":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5757,25.0024]}


                if dict_data['stationId'] == "C0AH40":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5771,25.1291]}
                if dict_data['stationId'] == "C0AH70":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5504,25.0487]}
                if dict_data['stationId'] == "C0AI40":#ok
                    dict_data['location'] = {'type':'Point','coordinates':[121.5132,25.1156]}
                    

            except:
                print('error: ',dict_data)

            self.__data_dict__ = dict_data
        except os.error as e:
            self.__data_dict__ = None


    def get_dict(self):
        return self.__data_dict__
