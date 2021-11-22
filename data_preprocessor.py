import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import os,sys
import datetime as dt
import ub_config as cfg


station_cat_list = pd.read_csv(cfg.ipython_path+"location_category.csv")

def uvi_deg2Category(df):
    #UVI_val_catgory = {'>30': 8, '21-30':7, '16-20':6, '11-15':5,'7-10':4, '3-6':3, '1-2':2,'<1':1,'0':0}
    UVI_catgory = {'uvi_30': 8, 'uvi_20_30': 7, 'uvi_16_20': 6, 'uvi_11_16': 5, 'uvi_7_11': 4, 'uvi_3_7': 3, 'uvi_1_3': 2, 'uvi_1': 1, 'uvi_0': 0}
    uvi_c = pd.Series(index = df.index,data=[None for _ in range(len(df.index))])
    uvi_c[df.UVI > 30] = 'uvi_30'
    idx = (df.UVI > 20) & (df.UVI <= 30)
    uvi_c[idx] = 'uvi_20_30'
    idx = (df.UVI > 16) & (df.UVI <= 20)
    uvi_c[idx] = 'uvi_16_20'
    idx = (df.UVI > 11) & (df.UVI <= 16)
    uvi_c[idx] = 'uvi_11_16'
    idx = (df.UVI > 7) & (df.UVI <= 11)
    uvi_c[idx] = 'uvi_7_11'
    idx = (df.UVI > 3) & (df.UVI <= 7)
    uvi_c[idx] = 'uvi_3_7'
    idx = (df.UVI > 1) & (df.UVI <= 3)
    uvi_c[idx] = 'uvi_1_3'
    uvi_c[df.UVI <= 1] = 'uvi_1'
    uvi_c[df.UVI == 0] = 'uvi_0'

    uvi_c = uvi_c.map(UVI_catgory) #turning to label encoding
    df.UVI = uvi_c

    return df

def wind_dir_deg2Compass(df):
    arr=["WD_N","WD_NNE","WD_NE","WD_ENE","WD_E","WD_ESE", "WD_SE", "WD_SSE","WD_S","WD_SSW","WD_SW","WD_WSW","WD_W","WD_WNW","WD_NW","WD_NNW"]
    idx = (((df.WDIR/22.5)+.5)% 16).astype(int)
    idx = idx.apply(lambda x: arr[x])
    df.WDIR = idx
    return df

def data_preprocess_web(df,ts_shift=False):

    try:
        emptyidx=[]
        
        try:
            for x in df.index:
                if df.time[x] is np.nan:
                    emptyidx = emptyidx + [x]
    
            if emptyidx:
                df = df.drop(index = emptyidx)
        except:
            pass
        
        # drop features
        df = df.drop(columns = ['SeaPres','GloblRad','CloudA','td','WSGust','WDGust'])
        
        
        # type transformation
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time'])

        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('H').asfreq()
    

        float_idx = ['station_id','HUMD','PRES', 'TEMP', 'WDIR', 'H_24R', 'WDSE', 'PrecpHour', 'UVI', 'Visb']
        df[float_idx] = df[float_idx].astype('float')


        # fill missing value
        df['sbi'] = df['sbi'].apply(lambda x: round(x, 0))

        
        fill_past_mean_tag = []
        interpolate_tag = ['TEMP','WDIR','H_24R','station_id','PRES','HUMD','WDSE']
        fillzero_tag = ['UVI','Visb'] #not just fill zero
        one_hot_tag = ['WDIR','weekday','hours']
        normalize_tag = ['HUMD','PRES', 'TEMP', 'H_24R', 'WDSE', 'Visb']

        for tag in fill_past_mean_tag:
            dfl = []
            ndf = df[tag]
            for month in range(0,11):
                x = month % 3
                #y = math.floor(month/3)
                data = ndf[ndf.index.month == (month+2)]
                idx = data.index[data.apply(np.isnan)]

                #get mean of each weekday
                meanss = []
                for wkday in range(0,7):
                      for hr in range(0,24):
                        means = round(data[(data.index.hour == hr)&(data.index.weekday == wkday)].mean())
                        meanss = meanss + [means]

                #replace na data
                for i in idx:
                    data.loc[i] = meanss[i.weekday()*23 + i.hour]

                dfl = dfl + [data]

            new_df = pd.concat(dfl)

            df[tag]= new_df.values

        for tag in interpolate_tag:
            df[tag] = df[tag].interpolate()

        for tag in fillzero_tag:
            idx_min = df.index.min()
            idx_max = df.index.max()
            i = 0
            new_df = df.UVI.astype(float)
            for day in pd.date_range(idx_min,idx_max,freq='D'):
                startdate = day + dt.timedelta(hours=5)
                enddate = day + dt.timedelta(hours=18)
                target_df = new_df[startdate:enddate]
                target_df2 = target_df.interpolate(limit_direction="both").copy()
                for i in target_df2.index:
                    df[tag].loc[i] = target_df2.loc[i]
            df[tag] = df[tag].fillna(0)

        
        df = uvi_deg2Category(df)
        df = wind_dir_deg2Compass(df)
        
        #add features
        df['weekday'] = df.index.weekday.astype(str)
        df.weekday = df.weekday.apply(lambda x: 'wkdy_' + x)
        df['hours'] = df.index.hour.astype(str)
        df.hours = df.hours.apply(lambda x: 'hrs_' +x)
        
        #one-hot encoding
        for tag in one_hot_tag:
            data_dum = pd.get_dummies(df[tag],sparse=True)
            end = pd.DataFrame(data_dum)
            df[end.columns] = end
            df = df.drop(columns=[tag])
           
        #normalization
        for tag in normalize_tag:
            df[tag] = (df[tag] - df[tag].min()) / (df[tag].max()-df[tag].min())
            
            
        # add holiday or not
        from workalendar.asia import Taiwan
        cal = Taiwan()
        holidayidx = []
        for t in cal.holidays(2020):
            for h in range(0,24):
                dd = dt.datetime.combine(t[0], dt.datetime.min.time())
                date_str = (dd + dt.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
                holidayidx = holidayidx + [date_str]
            
        for t in cal.holidays(2021):
            for h in range(0,24):
                dd = dt.datetime.combine(t[0], dt.datetime.min.time())
                date_str = (dd + dt.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
                holidayidx = holidayidx + [date_str]
            

        df['holiday'] = df.index.isin(holidayidx)

        ##df = pd.merge(df, station_cat_list, left_on='station_id', right_on='sno')

        for tag in ['sbi']:

            for i in range(1,13):
                df[tag+'_'+str(i)+'h'] = df[tag].shift(i)

            for i in range(1,8):
                df[tag+'_'+str(i)+'d'] = df[tag].shift(i*24)

            for tag in ['sbi']:
                df['y_' + tag] = df[tag].shift(-1)


        if ts_shift:
            ndf = df
            ndf['predict_hour'] = 1
            for tag in ['sbi']:
                ndf['y_' + tag] = df[tag].shift(-1)

            for i in range(1,13):
                ndf2 = df
                ndf2['predict_hour'] = i
                for tag in ['sbi']:
                    ndf2['y_' + tag] = df[tag].shift(-i)
                    #ndf2.to_csv("csvfile/parsed/" + "new_" + newfilename +"_predict_"+tag+"_"+str(i)+"h.csv")
                ndf = ndf.append(ndf2,ignore_index=True)

            ndf = ndf.dropna()
            ndf["time"] = pd.to_datetime(ndf["time"], format="%Y-%m-%d %H:%M")
            ndf.index = ndf["time"]
            
            df = ndf #for time shift use 

    except Exception as e:
        print('ERROR:',e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return df


if __name__ == '__main__':

    filepath = cfg.csv_merged_db_web_path
    filesinpath = os.listdir(filepath)
    for f in sorted(filesinpath): #for each file, ran model
        print("file name:", f)
        df = data_preprocess_web(pd.read_csv(filepath + f))
        break

    df.to_csv('test.csv')