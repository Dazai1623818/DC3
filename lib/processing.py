import pandas as pd
from datetime import datetime
import geopandas
import numpy as np

path = 'data/baseline/'

def extract_dates(df):
    min_date = df['date'].min().timestamp()
    max_date = df['date'].max().timestamp()
    min_date = datetime.fromtimestamp(min_date)
    max_date = datetime.fromtimestamp(max_date)
    return(min_date, max_date)

def slice_dates(df, date):
    return (df
     [df['date'] >= date[0]]
     [df['date'] < date[1]])   

def preprocess():
    df_conflict = pd.read_csv(f'{path}conflict.csv')
    df_admissions = pd.read_csv(f'{path}admissions.csv', parse_dates=['date'])
    df_admissions['district'] = df_admissions['district'].replace('Ceel Dheere', 'Ceel Dheer')
    return df_admissions


def map_counts(df):
    df = df.groupby(['district']).sum()
    counts = df['MAM_admissions']
    return np.sqrt(counts, out=np.zeros_like(counts), where=(counts!=0))

def plot_explore(counts):
    gdf = geopandas.read_file('data/maps/districts.geojson').drop(['date', 'validOn', 'ValidTo'], axis=1)
    gdf = (gdf
        [['admin2Name', 'admin1Name', 'geometry',
            'Shape_Leng', 'Shape_Area']])

    gdf['counts'] = (gdf
                    ['admin2Name']
                    .map(counts))
    
    return gdf.explore(column='counts',
                  legend_kwds=dict(loc='center left'), 
                  legend=True, 
                  cmap='OrRd', 
                  missing_kwds={"color": "black"})
    