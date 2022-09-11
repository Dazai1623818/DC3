import streamlit as st
import geopandas
from streamlit_folium import st_folium

district_map = geopandas.read_file('maps/districts.geojson').drop(['date', 'validOn', 'ValidTo'], axis=1)
region_map = geopandas.read_file('maps/regions.geojson')



map_type = st.radio(
     "Map Type",
     ('Districts', 'Regions'))

if map_type == 'Districts':
     st_folium(district_map.explore())
else:
     st_folium(region_map.explore())




