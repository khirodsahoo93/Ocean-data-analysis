
from numpy import pi, sin, cos,pi
import plotly.express as px
import datetime
import plotly.graph_objects as go
from time import sleep
from tqdm import tqdm
from random import randint
from ooipy.request import hydrophone_request
from ooipy.tools import ooiplotlib as ooiplt
import os, json
import pandas as pd
import numpy as np
import datetime
import ooipy
import matplotlib.pyplot as plt
from datetime import timedelta
import time
import warnings
warnings.filterwarnings("ignore")
def choose_df(df,flag,verbose=True):
    ship_cols=['MMSI', 'SHIPNAME', 'VESSEL TYPE','SPEED (KNOTSx10)','COURSE', 'HEADING', 'TIMESTAMP UTC',
       'LENGTH', 'Year','ship_Loc','LAT','LON']
    if flag==1:
        cols=ship_cols + ['distance(in km) axial','axial_Loc']
        df=df[cols].rename(columns={'distance(in km) axial':'distance(in km)'})
    elif flag==2:
        cols=ship_cols + ['distance(in km) central cald','central_caldera_Loc']
        df=df[cols].rename(columns={'distance(in km) central cald':'distance(in km)'})
    elif flag==3:
        cols=ship_cols + ['distance(in km) eastern cald','eastern_caldera_Loc']
        df=df[cols].rename(columns={'distance(in km) eastern cald':'distance(in km)'})
    if verbose==True:
        print(' Max distance: {} and Min distance: {}'.format(df['distance(in km)'].max(),df['distance(in km)'].min()))
    return df

def get_isolated_ships(df,rad,out_rad,min_d,verbose=True): #optimised version
  
    vessels=df[['MMSI','VESSEL TYPE']].drop_duplicates(subset=['MMSI'])
    df=df.sort_values(by=['TIMESTAMP UTC'],ascending=True)
    df_temp=df[df['distance(in km)']<(out_rad+1)] #find all ships inside radius = rad +1
    df_temp['rad_flag']=np.where(df_temp['distance(in km)'] <rad, 1, 0) #mark ships as 1 which are within radius=rad , otherwise 0
    df_temp['prev_MMSI']= df_temp['MMSI'].shift(1) # this will track when a new ship comes in
    df_temp['prev_rad_flag']= df_temp['rad_flag'].shift(1) # this will track when ship goes outside radius=rad
    count=0
    df_temp[['prev_MMSI','prev_rad_flag']].dropna(axis=0,inplace=True)
    #df_temp['break']=np.where((df_temp['MMSI']==df_temp['prev_MMSI']) & (df_temp['prev_rad_flag']==df_temp['rad_flag']),count,count+=1)
    df_temp['not_same_ship_as_prev']=~(df_temp['MMSI']==df_temp['prev_MMSI'])
    df_temp['xor_rad_flag_with_prev']=[bool(x)^bool(y) for x,y in zip(df_temp['rad_flag'],df_temp['prev_rad_flag'])]
    df_temp['or_ship_and_flag']=df_temp['not_same_ship_as_prev'] | df_temp['xor_rad_flag_with_prev']
    df_temp['break']=df_temp['or_ship_and_flag'].cumsum() # create partitions when a ship is continuously within radius=rad
    #df_temp_copy=df_temp.copy()
    #df_temp=df_temp.assign(flag=(df_temp['MMSI']==df_temp['prev_MMSI']) & (df_temp['prev_rad_flag']==df_temp['rad_flag']).cumsum())
    # df_temp.to_csv('ships_smoothpath.csv')
    df_temp=df_temp[df_temp['rad_flag']==1]
    df_temp_grp=df_temp.groupby(by=['MMSI','break']).agg({'TIMESTAMP UTC':['min','max','count']}).reset_index()
    df_temp_grp.columns=['MMSI','break','time_min','time_max','count']
    df_temp_grp.drop(df_temp_grp[df_temp_grp['time_max'] == df_temp_grp['time_min']].index,inplace=True)#remove ships which barely touched the radius
    #df_temp_grp.drop(df_temp_grp[df_temp_grp['count']<(min_t +1)].index,inplace=True) #filter ships with less than minimum number of timestamps
    df_temp_grp=df_temp_grp.merge(vessels,how='left',on='MMSI')
    
    df_temp_grp['len_of_recording']=(df_temp_grp['time_max']-df_temp_grp['time_min'])
    df_temp_grp['len_of_recording']=[x.total_seconds()/60.0 for x in df_temp_grp['len_of_recording']]
    df_temp_grp.drop(df_temp_grp[df_temp_grp['len_of_recording']<min_d].index,inplace=True)
    df_temp_grp.reset_index(inplace=True,drop=True)
    df_temp_grp.drop('break',axis=1,inplace=True)
    return df_temp_grp   

#Check status of ships' lonely timeframe in the overall ais dataset
def check_status_in_ais(ais,MMSI,min_time,max_time,rad):
    print('Min time : ', min_time)
    print('Max time : ', max_time)
    ais_test= ais[(ais['TIMESTAMP UTC']>min_time) & (ais['TIMESTAMP UTC']<max_time)]
    ais_test_lt=ais_test[ais_test['distance(in km)']<=rad]
    ais_test_gt=ais_test[ais_test['distance(in km)']>rad]
    n=ais_test_lt['MMSI'].nunique()
    min_distance_outside_rad=ais_test_gt['distance(in km)'].min()
    print('The number of unique lonely ships within the radius in the timeframe : ', n)
    print('The minimum distance of all other ships in the timeframe is : ',min_distance_outside_rad)
    return ais_test



def simp_spectrogram(hydrophone_idx,start_time,end_time,min_freq=None,max_freq=None):
    print('Start time : {} and End time : {}'.format(start_time,end_time))
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'

    time_diff=end_time-start_time
    time_diff=time_diff.total_seconds()/60.0 
    if time_diff >10:
        end_time=start_time + timedelta(minutes=10)
    else:
        end_time=end_time
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node,fmin=min_freq,fmax=max_freq, verbose=True, zero_mean=True)
    print(data_trace)
    if data_trace==None:
        print('data trace is none. Continuing to next')
        pass
    else:
        spec = data_trace.compute_spectrogram(L = 256,avg_time=None, overlap=0.9)
        #spec.compute_psd_welch()

        print('/************************************************************************************************/')
        ooipy.plot(spec, fmin=min_freq, fmax=max_freq, xlabel_rot=30,vmax=110) #xlabel changed from 30 to 10
        #ooipy.tools.ooiplotlib.plot_spectrogram(spec)
        #plt.xlim([0, 10])
        plt.show()

def get_acoustic(hydrophone_idx,start_time,end_time,min_freq=None,max_freq=None):
    #print('Start time : {} and End time : {}'.format(start_time,end_time))
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node,fmin=min_freq,fmax=max_freq, verbose=False, zero_mean=True)
    return data_trace


def get_spectogram(hydrophone_idx,lone_ships,num=5,ideal_dur=10): #ideal_duration -> ideal duration in minutes
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('plotting of spectrogram started')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Total number of ships :',len(lone_ships))
    if len(lone_ships)<num:
        num=len(lone_ships)
    print('Displaying spectrogram for {} ships'.format(len(lone_ships)))
    for i in range(num):
        list_dur=lone_ships.iloc[i]
        min_time=list_dur.time_min
        min_time=datetime.datetime(min_time.year,min_time.month,min_time.day,min_time.hour,min_time.minute,0)
        #min_time=min_time.strftime('%Y-%m-%d %H:%M')
        #min_time=datetime.datetime.strptime(min_time,'%Y-%m-%d %H:%M')
        print(min_time)
        max_time=list_dur.time_max
        max_time=datetime.datetime(max_time.year,max_time.month,max_time.day,max_time.hour,max_time.minute,0)
        #max_time=max_time.strftime('%Y-%m-%d %H:%M')
        #max_time=datetime.datetime.strptime(max_time,'%Y-%m-%d %H:%M')
        start_time = min_time
        end_time=min_time+timedelta(minutes=ideal_dur)
        if end_time > max_time:
            end_time=max_time
        print('{}. MMSI: {} , VESSEL TYPE: {}'.format(i+1,lone_ships['MMSI'].iloc[i],lone_ships['VESSEL TYPE'].iloc[i]))
        print('Start time : {} and End time : {}'.format(start_time,end_time))
        simp_spectrogram(hydrophone_idx,start_time,end_time)
        #alter_spectrogram_func(start_time,end_time)
        #time.sleep(20)

def get_circle_coordinates(rad,lat,lon):
    R = rad #in meters

    center_lon = lon
    center_lat = lat

    t = np.linspace(0, 2*pi, 100)

    circle_lon =center_lon + R*cos(t)
    circle_lat =center_lat +  R*sin(t)


    coords=[]
    lat1=[]
    lon1=[]
    for lo, la in zip(list(circle_lon), list(circle_lat)):
        lat1.append(la)
        lon1.append(lo)
        coords.append([lo, la]) 

    ### Another way to find coordinates of a circle
    N = 100 # number of discrete sample points to be generated along the circle

    # generate points
    circlePoints = []
    lat=[]
    lon=[]
    for k in range(N):
        # compute
        angle = pi*2*k/N
        dx = R*cos(angle)
        dy = R*sin(angle)
        point = {}
        la=center_lat + (180/pi)*(dy/6378137)
        lo=center_lon + (180/pi)*(dx/6378137)/cos(center_lat*pi/180)
        lat.append(la)
        lon.append(lo)
        # add to list
        circlePoints.append([lo,la])

    return lat,lon    

def get_map_plot(fn,df,rad1,inner_rad2,outer_rad2,lat,lon,time=None):
    
    df1 = df[(df['distance(in km)']<rad1)]
    df2 = df[(df['distance(in km)']>inner_rad2) & (df['distance(in km)']<outer_rad2) ]
    df3=pd.concat([df1,df2])

    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    outer_rad2=outer_rad2*1000
    if fn=='scatter':

   
        fig = px.scatter_mapbox(df3,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")
        
    elif fn=='density':
        fig = px.density_mapbox(df3,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
                    
      
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)
    lat_arr3,lon_arr3=get_circle_coordinates(outer_rad2,lat,lon)


    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6=px.scatter_mapbox(lat=lat_arr3,lon=lon_arr3,mapbox_style='carto-positron')
    fig6.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    fig.add_trace(fig6.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
                bearing=10,
                center=dict(
                    lat=lat,
                    lon=lon,
                )))
    
    fig.show()
    return df2

def get_isolated_map_plot(fn,df,rad1,inner_rad2,lat,lon,min_d):

    ais_test=pd.DataFrame()

    print('Finding lonely ships')
    iso_ships=get_isolated_ships(df,rad1,inner_rad2,min_d)
    print('Found lonely ships')

    print('Creating the dataframe with all ships and their locations in the lonely timeframes')
   
    
    for i in tqdm(range(len(iso_ships))):
        min_time=iso_ships.time_min[i]
        max_time=iso_ships.time_max[i]
        ais_test=pd.concat([ais_test,df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]])
    
    # df1 = df[(df['distance(in km) axial']<rad1)]
    # df2 = df[(df['distance(in km) axial']>inner_rad2) & (df['distance(in km) axial']<outer_rad2)]

    #ais_test= df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    print('Starting to plot now')
    if fn=='scatter':
        fig = px.scatter_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")

    elif fn=='density':
        fig = px.density_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
    
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)



    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6 = px.line_mapbox( lat=[lat,lat_arr1[0]], lon=[lon,lon_arr1[0]],hover_name=[rad1,rad1])
    fig7 = px.line_mapbox( lat=[lat,lat_arr2[10]], lon=[lon,lon_arr2[10]],hover_name=[inner_rad2,inner_rad2])
    fig6.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig7.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig8 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr1[0]],
    lon = [lon,lon_arr1[0]],
    marker = {'size': 4,'color':'black'}))
    fig9 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr2[10]],
    lon = [lon,lon_arr2[10]],
    marker = {'size': 4,'color':'black'}))

    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    # fig.add_trace(fig6.data[0])
    # fig.add_trace(fig7.data[0])
    fig.add_trace(fig8.data[0])
    fig.add_trace(fig9.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
            bearing=0,
            center=dict(
                lat=lat,
                lon=lon,
            )))

    fig.show()

def get_single_isolated_map_plot(hydrophone_idx,fn,df,rad1,inner_rad2,lat,lon,min_d,fmin,fmax):

    

    print('Finding lonely ships')
    iso_ships=get_isolated_ships(df,rad1,inner_rad2,min_d)
    print('Found lonely ships')
    
   
    
    i=randint(0,len(iso_ships))
    print('showing for MMSI: ',iso_ships.MMSI[i])
    min_time=iso_ships.time_min[i]
    max_time=iso_ships.time_max[i]
    ais_test=df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    #check_status_in_ais(ziggly2.MMSI[i],min_time,max_time,rad1)
    simp_spectrogram(hydrophone_idx,min_time,max_time,fmin,fmax)
    
    # df1 = df[(df['distance(in km) axial']<rad1)]
    # df2 = df[(df['distance(in km) axial']>inner_rad2) & (df['distance(in km) axial']<outer_rad2)]

    #ais_test= df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    print('Starting to plot now')
    if fn=='scatter':
        fig = px.scatter_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")

    elif fn=='density':
        fig = px.density_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
    
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)



    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6 = px.line_mapbox( lat=[lat,lat_arr1[0]], lon=[lon,lon_arr1[0]],hover_name=[rad1,rad1])
    fig7 = px.line_mapbox( lat=[lat,lat_arr2[10]], lon=[lon,lon_arr2[10]],hover_name=[inner_rad2,inner_rad2])
    fig6.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig7.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig8 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr1[0]],
    lon = [lon,lon_arr1[0]],
    marker = {'size': 4,'color':'black'}))
    fig9 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr2[10]],
    lon = [lon,lon_arr2[10]],
    marker = {'size': 4,'color':'black'}))

    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    # fig.add_trace(fig6.data[0])
    # fig.add_trace(fig7.data[0])
    fig.add_trace(fig8.data[0])
    fig.add_trace(fig9.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
            bearing=0,
            center=dict(
                lat=lat,
                lon=lon,
            )))

    fig.show()

def isolated_ais(ais,iso_ships,inner_rad):
    data=pd.DataFrame()
    for i in tqdm(range(len(iso_ships))):
        min_time=iso_ships.time_min[i]
        max_time=iso_ships.time_max[i]
        temp=ais[(ais['TIMESTAMP UTC']>=min_time) & (ais['TIMESTAMP UTC']<=max_time) & (ais['distance(in km)']<inner_rad)]
        temp['lonely_ship_idx']=i
        data=pd.concat([data,temp])
    return data