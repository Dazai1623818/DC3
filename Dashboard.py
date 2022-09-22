import geopandas
import pandas as pd
import streamlit as st
from lib import processing
from streamlit_folium import st_folium


st.set_page_config(layout="wide")
st.title('Somalia prevalence indicator visualization dashboard')
st.sidebar.button('sample button')

# Extract scale for slider
df = processing.preprocess() 
min_date, max_date = processing.extract_dates(df) 


# Input widget
date = st.slider(
     "Select timeframe:",
     value=(min_date, max_date))

# Slider output
processing.slice_dates(df, date)

# mapping variable to district name
df = df.groupby(['district']).sum()
counts = df['MAM_admissions'].to_dict()
# counts = (df['district']
#           .value_counts()
#           .to_dict())

gdf = geopandas.read_file('data/maps/districts.geojson').drop(['date', 'validOn', 'ValidTo'], axis=1)
gdf = (gdf
       [['admin2Name', 'admin1Name', 'geometry',
         'Shape_Leng', 'Shape_Area']])
gdf['counts'] = (gdf
                 ['admin2Name']
                 .map(counts)
                 .fillna(0))
st.write(gdf)

# Graph parameters
gdf = gdf.explore(column='counts',
                  cmap='Reds',
                  scheme='user_defined',
                  classification_kwds={'bins':counts.quantile([.25, .5, .75])})

st_folium(gdf, 
          width=2000, 
          height=900)




