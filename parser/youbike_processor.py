#
#   filename: youbike_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#   todo: you can parse the item you want only to the dict
#

import os
import gzip
import json
import datetime
from base_processor import base_processor

class YoubikeProcessor(base_processor):
    """
        Description of Youbike processor
        input:
         - filename
         - type_gzip: gzip or json
    """
    __type__ = 'Youbike Processor'

    def __init__(self):
        super().__init__()

    def read(self,filename,type_gzip=False):
        try:
            if type_gzip:
                ubike_f = gzip.open(filename, 'r')
                ubike_jdata = ubike_f.read()
                ubike_f.close()
                dict_data = json.loads(ubike_jdata.decode('utf-8'))
            else:
                json_file = open(filename, encoding="utf-8")
                dict_data = json.load(json_file)
                json_file.close()

            dict_tmp = []
            for key,value in dict_data['retVal'].items():
                if int(value['sno']) == 988:
                    continue;
                try:
                    value['sno'] = int(value['sno'])
                    value['tot'] = int(value['tot'])
                    value['sbi'] = int(value['sbi'])
                    value['bemp'] = int(value['bemp'])
                    value['lat'] = float(value['lat'])
                    value['lng'] = float(value['lng'])
                    value['act'] = int(value['act'])
                    value['mday'] = datetime.datetime.strptime(value['mday'], "%Y%m%d%H%M%S")
                    dict_tmp = dict_tmp +[value]
                except Exception as e:
                    print(e)
                    #print('error: ',value)

            self.__data_dict__ = dict_tmp

        except os.error as e:
            print('OS Error:', e)
            self.__data_dict__ = None



    def get_dict(self):
        return self.__data_dict__
