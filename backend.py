import json
import datetime
import sys

sys.path.append("Refactor3/parser/")
sys.path.append("parser/")
from youbike_processor import YoubikeProcessor
from weather_processor import WeatherProcessor

class weather_station(object):

    weather_id = None
    weather_name = None
    coordinate = []
    cur_dt = None

    info = {
        'temperature' : None,
        'humidity': None,
        'rainfall': None,
        'pressure': None,
        'wind_speed' : None,
        'wind_direction': None,
        'uvi':None,
        'precphour':None,
        'visibility':None,
        'max_wind_speed':None,
        'max_wind_direction':None
    }

    def __init__(self, wdict):
        self.weather_id = wdict['stationId']
        self.weather_name = wdict['locationName']
        self.coordinate = [wdict['lat'],wdict['lon']]
        self.cur_dt = wdict['time']

        self.info['temperature'] = wdict['TEMP']
        self.info['humidity'] = wdict['HUMD']
        self.info['pressure'] = wdict['PRES']
        self.info['wind_speed'] = wdict['WDSE']
        self.info['wind_direction'] = wdict['WDIR']
        self.info['max_wind_speed'] = wdict['WSGust']
        self.info['max_wind_direction'] = wdict['WDGust']

        if self.weather_id in ['466910','466920','466930']:
            self.info['rainfall'] = wdict['24R']
            self.info['precphour'] = wdict['PrecpHour']
            self.info['uvi'] = wdict['UVI']
            self.info['visibility'] = wdict['Visb']
        else:
            self.info['rainfall'] = wdict['H_24R']


    def set_weather_info(self, wdict):

        self.info['temperature'] = wdict['TEMP']
        self.info['humidity'] = wdict['HUMD']
        self.info['pressure'] = wdict['PRES']
        self.info['wind_speed'] = wdict['WDSE']
        self.info['wind_direction'] = wdict['WDIR']
        self.info['max_wind_speed'] = wdict['WSGust']
        self.info['max_wind_direction'] = wdict['WDGust']

        if self.weather_id in ['466910','466920','466930']:
            self.info['rainfall'] = wdict['24R']
            self.info['precphour'] = wdict['PrecpHour']
            self.info['uvi'] = wdict['UVI']
            self.info['visibility'] = wdict['Visb']
        else:
            self.info['rainfall'] = wdict['H_24R']

    def get_weather_info(self):
        return self.info

    def get_weather_info_by_name(self,name):
        return self.info[name]

class bike_station(object):

    sno = None
    name = None
    addr = None
    coordinate = []

    wid1 = None
    wid2 = None
    #weather_info = None

    tot = None
    sbi = None
    bemp = None
    act = False
    cur_dt = None

    bemp_model = None #read from pkl
    sbi_model = None #read from pkl

    def __init__(self, bdict):
        self.sno = int(bdict['sno'])
        self.name = bdict['sna']
        self.addr = bdict['ar']
        self.coordinate = [float(bdict['lat']), float(bdict['lng'])]
        self.tot = int(bdict['tot'])
        self.sbi = int(bdict['sbi'])
        self.bemp = int(bdict['bemp'])
        self.act = bool(bdict['act'])
        self.cur_dt = bdict['mday']  #datetime.datetime.strptime(bdict['mday'], "%Y%m%d%H%M%S")


    def update_realtime_data(self, active, cur_bemp, cur_sbi,current_datetime):
        # Update weather information inside
        # Predict bemp and sbi here
        self.sbi = int(cur_sbi)
        self.bemp = int(cur_bemp)
        self.act = bool(active)
        print(type(current_datetime))
        self.cur_dt = datetime.datetime.strptime(current_datetime, "%Y-%m-%d %H:%M:%S")

    def get_status(self):
        # Return dict with(active, cur_bemp, cur_sbi,current_datetime)
        st_status = {}
        st_status['act'] = self.act
        st_status['bemp'] = self.bemp
        st_status['sbi'] = self.sbi
        st_status['date'] = self.cur_dt
        return st_status

    def get_information(self):         #return dict with(sno, name, address,coordinate)
        st_info = {}
        st_info['sno'] = self.sno
        st_info['name'] = self.name
        st_info['addr'] = self.addr
        st_info['coordinate'] = self.coordinate
        st_info['tot'] =self.tot
        return st_info

    def get_weather_ids(self):
        return self.wid1, self.wid2

    def set_weather_ids(self,wid1, wid2):
        self.wid1 = wid1
        self.wid2 = wid2

    def set_weather_info(self,wdict):
        self.weather_info = wdict

    def get_weather_info(self):
        return self.weather_info

    def get_predict_bemp(self,index):
        pass
    def get_predict_sbi(self,index):
        pass
    def get_predict_bemp_all(self):
        pass
    def get_predict_sbi_all(self):
        pass

class backend(object):

    """app backend"""
    rawpath = "raw_data/"
    bike_raw_file  = "YouBikeTPNow.json"
    w_raw_file = "WeatherDataNow.json"
    w2_raw_file = "WeatherDataNow2.json"

    station_num = 0
    all_station = {}

    wstation1_num = 0
    wstation2_num = 0

    wstations1 = {}
    wstations2 = {}

    bike_parser = None
    weather_parser = None
    weather_parser2 = None

    def __init__(self):
        self.bike_parser = YoubikeProcessor()
        self.weather_parser = WeatherProcessor()
        self.weather_parser2 = WeatherProcessor()

    def cal_dist(self,coor1,coor2):
        return (coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2

    def set_wstation_by_bikeid(self,station_id):
        min_dist = 9999
        min_station = None

        bs = self.all_station[station_id]
        for wid in self.wstations1:
            ws = self.wstations1[wid]
            dist = self.cal_dist(bs.coordinate, ws.coordinate)
            if dist < min_dist:
                min_station = ws.weather_id
                min_dist = dist

        min_dist2 = 9999
        min_station2 = None
        for wid in self.wstations2:
            ws = self.wstations2[wid]
            dist = self.cal_dist(bs.coordinate,ws.coordinate)
            if dist < min_dist2:
                min_station2 = ws.weather_id
                min_dist2 = dist

        bs.set_weather_ids(min_station,min_station2)

        winfo = self.wstations1[min_station].get_weather_info()
        winfo2 = self.wstations2[min_station2].get_weather_info()
        winfo['precphour'] = winfo2['precphour']
        winfo['uvi'] = winfo2['uvi']
        winfo['visibility'] = winfo2['visibility']
        bs.set_weather_info(winfo)

    def update_all_station(self):
        #update weather information first
        self.weather_parser.read(self.rawpath + self.w_raw_file)
        wdata = self.weather_parser.get_dict()
        if not self.wstations1: #if empty
            for wd in wdata:
                ws = weather_station(wd)
                self.wstations1[ws.weather_id] = ws
        else:
            for wd in wdata:
                ws = self.wstations1[wd['stationId']]
                ws.set_weather_info(wd)
        # weater station 2
        self.weather_parser2.read(self.rawpath + self.w2_raw_file)
        wdata2 = self.weather_parser2.get_dict()
        if not self.wstations2:
            for wd in wdata2:
                ws = weather_station(wd)
                self.wstations2[ws.weather_id] = ws
        else:
            for wd in wdata2:
                ws = self.wstations2[wd['stationId']]
                ws.set_weather_info(wd)

        self.wstation1_num = len(self.wstations1)
        self.wstation2_num = len(self.wstations2)

        #json_file = open(self.rawpath + self.bike_raw_file, encoding='utf-8')
        self.bike_parser.read(self.rawpath + self.bike_raw_file)
        ubike_data = self.bike_parser.get_dict()
        if not self.all_station:
            #generate new bike statoin class for each station
            bs_dict = {}
            for value in ubike_data:
                bs_dict = value
                bs = bike_station(bs_dict)
                self.all_station[int(value['sno'])] = bs
            self.station_num = len(ubike_data)
            # find nearest weather station for each bike station
            for key in self.all_station:
                bs = self.all_station[key]
                self.set_wstation_by_bikeid(bs.sno)
        else:
            #find exist station and update
            for value in ubike_data:
                sno = value['sno']
                bs = self.all_station[sno]
                bs.update_realtime_data(value['act'], value['bemp'], value['sbi'],value['mday'])

    def get_station_number(self):
        return self.station_num

    def get_all_station_list(self):
        #station_list = []
        #for key,val in self.all_station:
         #   station_list = station_list + [{key:val}]
        #print(self.all_station)
        return self.all_station #station_list

    def get_weather_number(self,first):
        if first:
            ret = self.wstation1_num
        else:
            ret = self.wstation2_num
        return ret

    def get_station_info(self,station_id):
        bs = self.all_station[station_id]
        return bs.get_information()

    def get_station_status(self,station_id):
        bs = self.all_station[station_id]
        return bs.get_status()

    def get_current_station_status(self,station_id):
        bs = self.all_station[station_id]
        return bs.get_status()

    def get_all_weather_id(self,first):
        wlist = []
        if first:
            for key in self.wstations1:
                wlist = wlist + [key]
        else:
            for key in self.wstations2:
                wlist = wlist + [key]
        return wlist

    def get_weather_by_station(self,station_id): #by bs
        try:
            ret = self.all_station[station_id].get_weather_info()
        except:
            ret = None

        return ret

    def get_preidct_by_station(self,station_id,bempflag):
        pass

    def get_predict_by_hour(self,hour,bempflg): #all station
        pass

if __name__ == "__main__":
    be = backend()
    be.update_all_station()

    for i in range(be.get_station_number()):
        try:
            print(be.get_station_info(i+1))
            print(be.get_current_station_status(i+1))
            ws = be.get_weather_by_station(i+1)
            print(ws)
        except:
            print("Station is not exist:", i+1)