import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium
from folium import plugins

data_1=[]
data_1=pd.read_csv('./5_w.csv',sep=';',encoding='cp1251')
#data_1.drop('Unnamed: 0',axis=1,inplace=True)
data_1['xy']=data_1.apply(lambda x:[x['Широта'],x['Долгота']],axis=1)
data_2=pd.read_csv('./53_w.csv',sep=';',encoding='cp1251')
#data_2.drop('Unnamed: 0',axis=1,inplace=True)
data_2['xy']=data_2.apply(lambda x:[x['Широта'],x['Долгота']],axis=1)
#display(data[['Широта','Долгота','Высота','xy']].head(3))

m = folium.Map (data_1.loc[0,'xy'], zoom_start = 13)
title_html = '''<h3 align="center" style="font-size:21px"><b>{}</b></h3>'''.format('Тест')
m.get_root().html.add_child(folium.Element(title_html))
route = folium.PolyLine (list(data_1['xy']),вес = 3, color = 'green', непрозрачность = 0.8).add_to (m)
route = folium.PolyLine (list(data_2['xy']),вес = 3, color = 'blue', непрозрачность = 0.8).add_to (m)
for i in range(len(data_1)):
    folium.CircleMarker(location=data_1.loc[i,'xy'], radius =2, popup=str(i),fill_color='red',color='red',fill_opacity = 0.9).add_to(m)
folium.CircleMarker(location=[55.431136,37.544997], radius =2, popup='Подольск',fill_color='red',color='red',fill_opacity = 0.9).add_to(m)
folium.CircleMarker(location=[55.755864,37.617698], radius =3, popup='Москва',fill_color='red',color='red',fill_opacity = 0.9).add_to(m)         
display(m)

lat1=55.431136
lon1=37.544997
lat2=55.755864
lon2=37.617698
lat1_rad = np.radians(lat1)
lon1_rad = np.radians(lon1)
lat2_rad = np.radians(lat2)
lon2_rad = np.radians(lon2)

dlon = lon2_rad - lon1_rad
 
y = np.sin(dlon) * np.cos(lat2_rad)
x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

az_rad = np.arctan2(y, x)
az_deg = np.degrees(az_rad)

# Normalize to [0, 360) degrees
az_deg = (az_deg + 360) % 360
we_rad=np.arctan((lat2_rad-lat1_rad)/(lon2_rad-lon1_rad))
we_deg=np.degrees(we_rad)
we_deg=(we_deg + 360) % 360

EARTH_RADIUS = 6371000  # in meters
# Haversine formula
dlat = lat2_rad - lat1_rad
dlon = lon2_rad - lon1_rad
a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat1_rad) *np.cos(lat2_rad) * np.sin(dlon/2) * np.sin(dlon/2)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
distance = EARTH_RADIUS * c
 
    
print('az_deg=',az_deg,' we_rad=',we_rad,' we_deg=',we_deg,' distance=',distance)
def azimut(x):
    lat1_rad = np.radians(x['Широта'])
    lon1_rad = np.radians(x['Долгота'])
    lat2_rad = np.radians(x['Широта_'])
    lon2_rad = np.radians(x['Долгота_'])
    dlon = lon2_rad - lon1_rad
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    az_rad = np.arctan2(y, x)
    az_deg = np.degrees(az_rad)
    # Normalize to [0, 360) degrees and return
    return((az_deg + 360) % 360)
def dist(x):
    lat1_rad = np.radians(x['Широта'])
    lon1_rad = np.radians(x['Долгота'])
    lat2_rad = np.radians(x['Широта_'])
    lon2_rad = np.radians(x['Долгота_'])
    EARTH_RADIUS = 6371000  # in meters
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat1_rad) *np.cos(lat2_rad) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = EARTH_RADIUS * c
    # Normalize to [0, 360) degrees and return
    return(EARTH_RADIUS * c)
data_1['Широта_']=data_1['Широта'].shift(-1)
data_1['Долгота_']=data_1['Долгота'].shift(-1)
data_1.loc[len(data_1)-1,'Широта_']=data_1.loc[len(data_1)-1,'Широта']
data_1.loc[len(data_1)-1,'Долгота_']=data_1.loc[len(data_1)-1,'Долгота']

data_1['azimut']=data_1.apply(lambda x:azimut(x),axis=1)
data_1['dist']=data_1.apply(lambda x:dist(x),axis=1)
data_1['s_a_v']=data_1.apply(lambda x: x['Скорость']+np.sqrt(x['Ускорение по оси Z']**2+x['Ускорение по оси X']**2)*9.81/2,axis=1)
display(data_1[['Широта','Широта_','Долгота','Долгота_','azimut','путь','dist','Скорость','Ускорение по оси X','Ускорение по оси Y','Ускорение по оси Z','s_a_v']][175:178])