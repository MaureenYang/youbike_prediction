import urllib.request
import time, threading
import datetime
import os, sys, shutil, codecs ,json
import pandas as pd
from bs4 import BeautifulSoup
import ub_config as cfg

wdata_from_web = True
realtime_data = False


ONE_MIN = 60
ONE_HOUR = ONE_MIN * 60
ONE_DAY = ONE_HOUR * 24


ubike_url = "https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json"
weather_url = "http://opendata.cwb.gov.tw/opendataapi?dataid=O-A0001-001&authorizationkey=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725&downloadType=WEB&format=JSON"
#weather2_url = "http://opendata.cwb.gov.tw/opendataapi?dataid=O-A0001-001&authorizationkey=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725"
weather2_url = "https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/O-A0003-001?Authorization=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725&downloadType=WEB&format=JSON"



def GetUbikeDataThread():

    while True:
        try:
            urllib.request.urlretrieve(ubike_url, "YouBikeTP.json")

            try:
               shutil.copyfile("YouBikeTP.json",cfg.raw_file_path + "YouBikeTPNow.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except Exception as e:
               print("Unexpected error:", sys.exc_info())
               print("error:",e)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "YouBikeTP_" + st + ".json"
            os.rename("YouBikeTP.json",fname)
            shutil.move(fname,cfg.raw_file_path + "youbike_data/")
            print("get file : "+fname)

            time.sleep(ONE_MIN)    #every minites
            
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Youbike HTTPerror]" + e)
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Youbike URLerror]" + e)
            time.sleep(10)
        except TimeoutError as e:
            print("[Youbike Timeouterror]" + e)
            time.sleep(10)
        except Exception as e:
            print("[Youbike]Unexpected Error: " + e)
            time.sleep(10)


def GetWeatherThread():

    while True:
        try:
            urllib.request.urlretrieve(weather_url,"weather_data.json")
            try:
               shutil.copyfile("weather_data.json",cfg.raw_file_path + "WeatherDataNow.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except:
               print("Unexpected error:", sys.exc_info())

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "weather_data_" + st + ".json"
            os.rename("weather_data.json",fname)
            shutil.move(fname,cfg.raw_file_path +"weather_data/")
            print("get file : "+fname)


            time.sleep(ONE_HOUR) #every hour

        except TimeoutError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather error]time out! try again " + e)
            time.sleep(30)
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather HTTP error] " + e)
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Weather URL error]" + e)
            time.sleep(10)

def GetWeather2Thread():
    while True:
        try:
            urllib.request.urlretrieve(weather2_url, "weather_data2.json")
            try:
               shutil.copyfile("weather_data2.json",cfg.raw_file_path +"WeatherDataNow2.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except:
               print("Unexpected error:", sys.exc_info())

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "weather2_data_" + st + ".json"
            os.rename("weather_data2.json",fname)
            shutil.move(fname,cfg.raw_file_path+"weather2_data/")
            print("get file : "+fname)
            time.sleep(ONE_HOUR) #every hour

        except TimeoutError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather error]time out! try again" + e)
            time.sleep(30)
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather HTTP error]" + e)
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Weather URL error]" + e)
            time.sleep(10)



## T 表示微量(小於 0.1mm)，x 表故障，& 代表降水量資料累積於後，V 表風向不定，/表不明，…表無觀測。
def float_trans(data):
 
    try:
        if 'T' in data:
            return 0
       
        if 'x' in data or 'V' in data or '/' in data or '...' in data:
            return None
        
        if '&' in data: #accumulate
            return 'a'
        
        f = float(data)
        return f
    except Exception as e:
        print(e,":", data)
        return None


def get_weather_from_web(date_start="20200501", date_end="20210612"):
    
    date_range = pd.date_range(date_start,date_end,freq='D')
    
    for key, val in cfg.weather_station_dict.items():
        print('station:',key,', ',val)
        try:
            for d in date_range:
                date_str = str(d.year)+"-"+str(d.month).zfill(2)+"-"+str(d.day).zfill(2)
                print(date_str,' ', val,' ',key)
                str_name = str(val['name'].encode('utf-8'))
                str_name = str_name[2:-1:]
                str_name = str_name.replace("\\x", "%25",10)
    
                url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station="+key+"&stname="+str_name+"&datepicker="+date_str+"#"
                html_filename = "weatherdata.html"
                
                urllib.request.urlretrieve(url, html_filename)
                f=codecs.open(html_filename, 'r',encoding="utf-8")
                html_string = f.read()
    
                soup = BeautifulSoup(html_string, 'html.parser')
                p = soup.html.find_all("td")
                k = 0
                for item in p:
                    if item.string == '01':
                        print('find hour!')
                        break
                    k = k+1
    
                p = p[k::]
                dict = {}
                for idx in range(0,len(p),17):
                    item = p[idx:idx+16]
                    hour = int(item[0].string) #hour
                    pres = float_trans(item[1].string) #PRES
                    SeaPres = float_trans(item[2].string)
                    temp = float_trans(item[3].string) #TEMP
                    Td = float_trans(item[4].string)
                    humd = float_trans(item[5].string) #HUMD
                    wdse = float_trans(item[6].string) #WDSE
                    wdir = float_trans(item[7].string) #WDIR
                    WSGust = float_trans(item[8].string) #WDIR
                    WDGust = float_trans(item[9].string) #WDIR
                    h_24r = float_trans(item[10].string) #H_24R
                    PrecpHour = float_trans(item[11].string) #H_24R
                    GloblRad = float_trans(item[12].string) #H_24R
                    Visb = float_trans(item[13].string) #H_24R
                    UVI = float_trans(item[14].string) #H_24R
                    CloudA = float_trans(item[15].string) #H_24R
    
                    dict['locationName']=val
                    dict['stationId']=key
                    dict['time'] = datetime.datetime.strftime(datetime.datetime(d.year, d.month, d.day, hour-1, 0, 0), '%Y-%m-%dT%H:%M:%S')
                    dict['SeaPres'] = SeaPres
                    dict['td'] = Td
                    dict['PRES'] = pres
                    dict['TEMP'] = temp
                    dict['HUMD'] = humd
                    dict['WDSE'] = wdse
                    dict['WDIR'] = wdir
                    dict['H_24R'] = h_24r
                    dict['WSGust'] = WSGust
                    dict['WDGust'] = WDGust
                    dict['PrecpHour'] = PrecpHour
                    dict['GloblRad'] = GloblRad
                    dict['Visb'] = Visb
                    dict['UVI'] = UVI
                    dict['CloudA'] = CloudA
                    date_str1 = date_str +'-'+ str(hour-1).zfill(2)+ "-00-00"
    
                    filename = cfg.raw_weather_web_path + "web_"+key +'_'+ date_str1 + ".json"
                    print('save to ',filename)
                    with open(filename, 'w') as fp:
                        json.dump(dict, fp)
                        
        except Exception as e:
            print(e)
            print('station:',key,', ',val)


#### main.py
if __name__ == "__main__":

    if wdata_from_web:
        get_weather_from_web()#(date_start="20201001", date_end="20201031")

    if realtime_data:
        print ("Crawler Starting...")  
        #create thread
        ubike_thread = threading.Thread(target = GetUbikeDataThread)
        weather_thread = threading.Thread(target = GetWeatherThread)
        weather2_thread = threading.Thread(target = GetWeather2Thread)
    
        ubike_thread.start()
        weather_thread.start()
        weather2_thread.start()
    
        ubike_thread.join()
        weather_thread.join()
        weather2_thread.join()
    
        print ("Crawler Finished")
