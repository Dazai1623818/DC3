import geopandas
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime
import mapclassify as mc


st.set_page_config(layout="wide")

district_map = geopandas.read_file('maps/districts.geojson').drop(['date', 'validOn', 'ValidTo'], axis=1)
region_map = geopandas.read_file('maps/regions.geojson')
df = pd.read_csv('data/conflict.csv')

# Extract scale for slider
min_date = datetime.strptime(df['date'].min(), '%Y-%m-%d')
max_date = datetime.strptime(df['date'].max(), '%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Input widgets
map_type = st.radio(
     "Map Type",
     ('districts', 'regions'))

date = st.slider(
     "Schedule your appointment:",
     value=(min_date, max_date))

# Slider output
df = (df
     [df['date'] >= date[0]]
     [df['date'] < date[1]])

# mapping number of conflicts to district name
counts = (df['district']
          .value_counts()
          .to_dict())
gdf = geopandas.read_file(f'maps/{map_type}.geojson')
gdf.set_index('admin2Name')
gdf['counts'] = (gdf
                 ['admin2Name']
                 .map(counts)
                 .fillna(0))


# Graph parameters
gdf = gdf.explore(column='counts',
                  cmap='Reds',
                  scheme='user_defined', 
                  classification_kwds={'bins':[0, 10, 20, 30, 40, 50, 61]})

st_folium(gdf, 
          width=1500, 
          height=1000)




