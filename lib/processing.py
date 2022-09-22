import pandas as pd
from datetime import datetime

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