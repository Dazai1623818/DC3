'''------------SECTION IMPORTS---------------------'''
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import theano
import theano.tensor as tt
import seaborn as sns
from tqdm import tqdm

'''------------SECTION USER VARIABLES--------------'''
#Define the path to your datafolder below
your_datapath = 'Data/'

'''------------SECTION FUNCTIONS--------------'''
#Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries
def make_district_df_semiyearly(datapath, district_name):
    """
    Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder
    district_name : string
        Name of the district

    Returns
    -------
    df : pandas dataframe
    """
    prevalence_df = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    prevalence_df = (prevalence_df
                     .query('district == @district_name')
                     .sort_values('date')
                    )
    
    covid_df = pd.read_csv(datapath + 'covid.csv', parse_dates=['date'])
    covid_df = (covid_df
                .groupby(pd.Grouper(key='date', freq='6M')).sum()
                .reset_index()
               )
    covid_df = (covid_df
                .assign(date=covid_df.date.apply(lambda x : x.replace(day=1)))
                .sort_values('date')
               )
    
    ipc_df = pd.read_csv(datapath + 'ipc.csv', parse_dates=['date'])
    ipc_df = (ipc_df
              .query('district == @district_name')
              .sort_values('date')
             )
    
    risk_df = pd.read_csv(datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    risk_df = (risk_df
               .query('district == @district_name')
               .groupby(pd.Grouper(key='date', freq='6M')).mean()
               .reset_index()
              )
    risk_df = (risk_df
               .assign(date=risk_df.date.apply(lambda x : x.replace(day=1)))
               .sort_values('date')
              )
    
    production_df = pd.read_csv(datapath + 'production.csv', parse_dates=['date'])
    production_df = (production_df
                     .query('district == @district_name')
                    )
    production_df = (production_df
                     .assign(cropdiv=production_df.count(axis=1))
                     .sort_values('date')
                    )

    #Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(
        left=pd.merge_asof(
            left=pd.merge_asof(
                left=pd.merge_asof(
                    left=prevalence_df, right=ipc_df, direction='backward', on='date'
                ), right=production_df, direction='backward', on='date'
            ), right=risk_df, direction='backward', on='date'
        ), right=covid_df, direction='backward', on='date'
    )  
    df = (df
          .assign(prevalence_6lag=df['GAM Prevalence'].shift(1),
                  next_prevalence=df['GAM Prevalence'].shift(-1),
                  month=df['date'].dt.month,
                  increase=[*[False if x[1]<x[0] else True for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))], False],
                  increase_numeric=[*[x[1] - x[0] for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))], 0],
                 )
          .rename(columns={'GAM Prevalence': 'prevalence',
                           'new_cases': 'covid',
                           'ndvi_score': 'ndvi',
                           'phase3plus_perc': 'ipc',
                           'total population': 'population'}
                 )
         )
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan #No info on next month
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan #No info on next month
    df.loc[(df.date < pd.to_datetime('2020-03-01')), 'covid'] = 0
    df = df[['date', 'district', 'prevalence', 'next_prevalence', 'prevalence_6lag', 'covid', 'ndvi', 
               'ipc', 'cropdiv', 'population', 'month', 'increase', 'increase_numeric']]
    
    return df
    
#Function that combines the semiyearly dataset (from the function make_district_df_semiyearly) of all districts
def make_combined_df_semiyearly(datapath):
    """
    Function that creates a pandas dataframe for all districts with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder

    Returns
    -------
    df : pandas dataframe
    """

    prevdf = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    districts = prevdf['district'].unique()
    
    df_list = [make_district_df_semiyearly(datapath, district).assign(district=district) for district in districts]        
    df = pd.concat(df_list, ignore_index=True)
    df = df.assign(district_encoded=df['district'].astype('category').cat.codes)

    return df

'''------------SECTION DATAFRAME CREATION--------------'''
#Create the dataframe for all districts
df = (make_combined_df_semiyearly(your_datapath)
      .dropna()
      .sort_values('date')
      .reset_index(drop=True)
     )

# Priors:

### $\phi=N(0, 20)$

### $\sigma=Exp(1)$

# Likelihood:

### $x_t|\phi_1, \sigma, x_1\sim N(\phi x_{t-1}, \sigma)$

# Posterior:

### $\phi, \sigma|x\sim ?$

# Explanation:

### $\sigma$ for generating $\phi_1$ is so high because the current distribution is unknown. Hence higher variance allows flatter distribution and have more space in exploring hyperparameters.

### Data generation formula uses data from the last time lag.

'''------------SECTION PROBABLISTIC MODEL--------------'''
district_name = 'Sablaale'
needed_data = df.query("district == @district_name")
prevalence_array = np.array(needed_data['prevalence_6lag'].tolist() + [
    needed_data['prevalence'].tolist()[-1], 
    needed_data['next_prevalence'].tolist()[-1]
])

# NUTS

with pm.Model() as bayes_model:
     coefs = pm.Normal("coefs", mu = 0, sigma = 20, size = 2)
     sigma = pm.Exponential("sigma", lam = 1)
    
     likelihood = pm.AR("x", coefs, sigma, observed = prevalence_array[: 7])
     trace = pm.sample(5000, cores = 2)
    
plt.figure(figsize = (7, 7))
pm.plot_trace(trace)
plt.tight_layout()

posterior = trace.posterior.stack(sample=['chain', 'draw'])

coef_1_vals = posterior['coefs'][0]
coef_2_vals = posterior['coefs'][1]
sigma_vals = posterior['sigma']

def plot_final_prediction(i, district):
    sample_size = 10000
    result = np.array([np.random.choice(coef_1_vals) * prevalence_array[i - 1] 
                       + np.random.choice(coef_2_vals) * prevalence_array[i - 2] 
                          + np.random.normal(0, np.random.choice(sigma_vals)) for _ in tqdm(range(sample_size))])
    ax = plt.figure(figsize = (10, 4))
    ax = sns.distplot(result)
    p1 = ax.axvline(result.mean(), color = 'r', linewidth = 1, label = 'Predicted Next Prevalence Mean')
    p2 = ax.axvline(prevalence_array[i], color = 'b', linewidth = 1, label = 'Actual Next Prevalence')
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.set_title(f"Next Prevalence for Record {i} in {district}: \n Actual Prevalence: 
                 {round(prevalence_array[i], 3)}, Predicted Prevalence Mean: 
                 {round(result.mean(), 3)}, Predicted Prevalence SD: {round(result.std(), 3)}")
                 
plot_final_prediction(5, district_name)
