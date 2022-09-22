import pandas as pd
import streamlit as st
from lib.processing import *
from streamlit_folium import st_folium


no_dates = RuntimeError('No events! Please select larger timeframe')
st.set_page_config(layout="wide")
st.title('Somalia prevalence indicator visualization dashboard')
st.sidebar.button('sample button')

# Extract scale for slider
df = preprocess() 
min_date, max_date = extract_dates(df) 

dataset = st.selectbox('Select dataset to display',
                       ('conflict', 'admissions'))
                       

# Input widget
date = st.slider(
     "Select timeframe:",
     value=(min_date, max_date))

# Slider output
df = slice_dates(df, date)

if len(df) < 1:
     st.warning(no_dates)
     st.stop()

# mapping variable to district name
counts = map_counts(df)

# Graph parameters
m = plot_explore(counts)
st_folium(m,
          height=900,
          width=1200)
