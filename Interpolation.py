'''------------SECTION IMPORTS---------------------'''
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance, plot_tree
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm

'''------------SECTION USER VARIABLES--------------'''
#Define the path to your datafolder below
your_datapath = 'Data/'

#Define search space for number of trees in random forest and depth of trees
num_trees_min = 1
num_trees_max = 128

depth_min = 1
depth_max = 20

'''------------SECTION FUNCTIONS--------------'''
#Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries
def interpolate(df):
    """
    Function that creates more data from the semiyearly entries to monthly entries using linear interpolation

    Parameters
    ----------
    df : pandas dataframe
        Dataframe from csv file that contains semiyearly data

    Returns
    -------
    df : pandas dataframe
    """    
    if df.shape[0] == 0:
        return df
    else:
        all_date = pd.date_range(start = df['date'].min(), end = df['date'].max(), freq = 'MS')
        df = (df
              .set_index("date")
              .reindex(all_date)
              .reset_index()
              .rename(columns = {"index": "date"})
              # since District column is constant we can 
              # forward fill on the Name column
              .assign(district=lambda x: x.district.ffill())
             )
        for column in [i for i in df.columns.tolist() if df.dtypes[i] == 'float64']:
            df[column] = df[column].interpolate(method = 'linear')
        return df

def make_district_df_monthly(datapath, district_name):
    """
    Function that creates a pandas dataframe for a single district with columns for the baseline model with monthly entries

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
    prevalence_df = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates = ['date'])
    prevalence_df = (prevalence_df
                     .drop(columns = ['Unnamed: 0'])
                     .query('district == @district_name')
                     .sort_values('date')
                     .pipe(interpolate)
                    )
    
    covid_df = pd.read_csv(datapath + 'covid.csv', parse_dates = ['date'])
    
    ipc_df = pd.read_csv(datapath + 'ipc.csv', parse_dates = ['date'])
    ipc_df = (ipc_df
              .query('district == @district_name')
              .sort_values('date')
              .pipe(interpolate)
             )
    
    risk_df = pd.read_csv(datapath + 'FSNAU_riskfactors.csv', parse_dates = ['date'])
    risk_df = (risk_df
               .query('district == @district_name')
               .sort_values('date')
               .pipe(interpolate)
              )
    
    production_df = pd.read_csv(datapath + 'production.csv', parse_dates = ['date'])
    production_df = (production_df
                     .query('district == @district_name')
                     .assign(cropdiv = lambda x: x.count(axis = 1))
                     .sort_values('date')
                     .pipe(interpolate)
                    )

    #Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(
        left=pd.merge_asof(
            left=pd.merge_asof(
                left=pd.merge_asof(
                    left=prevalence_df, right=ipc_df, direction='backward', on='date', suffixes=('', '_remove')
                ), right=production_df, direction='backward', on='date', suffixes=('', '_remove')
            ), right=risk_df, direction='backward', on='date', suffixes=('', '_remove')
        ), right=covid_df, direction='backward', on='date', suffixes=('', '_remove')
    )  
    df = (df
          .drop([i for i in df.columns if 'remove' in i], axis=1)
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
    
#Function that combines the semiyearly dataset into monthly dataset (from the function make_district_df_monthly) of all districts
def make_combined_df_monthly(datapath):
    """
    Function that creates a pandas dataframe for all districts with columns for the baseline model with monthly entries

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
    
    df_list = [make_district_df_monthly(datapath, district).assign(district=district) for district in districts]        
    df = pd.concat(df_list, ignore_index=True)
    df = df.assign(district_encoded=df['district'].astype('category').cat.codes)

    return df
  
  '''------------SECTION DATAFRAME CREATION--------------'''
#Create the dataframe for all districts
df = (make_combined_df_monthly(your_datapath)
      .dropna()
      .sort_values('date')
      .reset_index(drop=True)
     )

'''------------SECTION XGBOOST FEATURE EXTRACTION--------------'''
#WARNING: this process can take some time, since there are a lot of hyperparameters to investigate. The search space can be manually reduced to speed up the process.

#Define target and explanatory variables
X = df.drop(columns = ['increase', 'increase_numeric', 'date', 'district', 'prevalence', 'next_prevalence']) #Note that these columns are dropped, the remaining columns are used as explanatory variables
y = df['next_prevalence'].values

#Peform evaluation on full data
Xtrain = X[: 33 * 5 * 6]
ytrain = y[: 33 * 5 * 6]
Xtest = X[33 * 5 * 6: ]
ytest = y[33 * 5 * 6: ]

# fit model no training data
model = XGBRegressor()
model.fit(Xtrain, ytrain)

# feature importance
print(model.feature_importances_)

# plot feature importance
plot_importance(model)
plt.show()

'''------------SECTION XGBOOST CROSS VALIDATION--------------'''
#WARNING: this process can take some time, since there are a lot of hyperparameters to investigate. The search space can be manually reduced to speed up the process.

#Create empty list to store model scores
parameter_scores = []

#Define target and explanatory variables
X = df.drop(columns = ['increase', 'increase_numeric', 'date', 'district', 'prevalence', 'next_prevalence']) #Note that these columns are dropped, the remaining columns are used as explanatory variables
y = df['next_prevalence'].values

#Define all possible column combinations with the most important features always included
important_feature = ['prevalence_6lag', 'ipc', 'ndvi', 'population', 'month', 'cropdiv', 'district_encoded', 'covid']
possible_columns_set = [important_feature[: len(important_feature) - i] for i in range(len(important_feature))]

#Investigate every subset of explanatory variables
for features in tqdm(possible_columns_set):

    internal_parameter_scores = []
    
    for num_trees in tqdm(range(num_trees_min, num_trees_max)):
    
        for depth in tqdm(range(depth_min, depth_max)):
        
            #First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
            Xtrain = X[: 33 * 3 * 6][features].copy().values
            ytrain = y[: 33 * 3 * 6]
            Xtest = X[33 * 3 * 6: 33 * 4 * 6][features].copy().values
            ytest = y[33 * 3 * 6: 33 * 4 * 6]

            #Create a RandomForestRegressor with the selected hyperparameters and random state 0.
            clf = XGBRegressor(n_estimators=num_trees, max_depth=depth, early_stopping_round=50)

            #Fit to the training data
            clf.fit(Xtrain, ytrain)

            #Make a prediction on the test data
            predictions = clf.predict(Xtest)

            #Calculate mean absolute error
            MAE1 = mean_absolute_error(ytest, predictions)


            #Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
            Xtrain = X[: 33 * 4 * 6][features].copy().values
            ytrain = y[: 33 * 4 * 6]
            Xtest = X[33 * 4 * 6: 33 * 5 * 6][features].copy().values
            ytest = y[33 * 4 * 6: 33 * 5 * 6]

            #Create a RandomForestRegressor with the selected hyperparameters and random state 0.
            clf = XGBRegressor(n_estimators=num_trees, max_depth=depth, early_stopping_round=50)

            #Fit to the training data
            clf.fit(Xtrain, ytrain)

            #Make a prediction on the test data
            predictions = clf.predict(Xtest)

            #Calculate mean absolute error
            MAE2 = mean_absolute_error(ytest, predictions)

            #Calculate the mean MAE over the two folds
            mean_MAE = (MAE1 + MAE2)/2

            #Store the mean MAE together with the used hyperparameters in list 
            internal_parameter_scores.append((mean_MAE, num_trees, depth))
        
    #Sort the models based on score and retrieve the hyperparameters of the best model
    internal_parameter_scores.sort(key=lambda x: x[0])
    parameter_scores.append((internal_parameter_scores[0][0], internal_parameter_scores[0][1], 
                            internal_parameter_scores[0][2], len(features)))
    
parameter_scores.sort(key=lambda x: x[0])
print(parameter_scores)

#Sort the models based on score and retrieve the hyperparameters of the best model
best_model_score = parameter_scores[0][0]
best_model_trees = parameter_scores[0][1]
best_model_depth = parameter_scores[0][2]
best_model_columns_nr = parameter_scores[0][3]



'''------------SECTION FINAL EVALUATION--------------'''
X = df[important_feature[: best_model_columns_nr]].values
y = df['next_prevalence'].values

#If there is only one explanatory variable, the values need to be reshaped for the model
if best_model_columns_nr == 1:
    X = X.reshape(-1, 1)

#Peform evaluation on full data
Xtrain = X[: 33 * 5 * 6]
ytrain = y[: 33 * 5 * 6]
Xtest = X[33 * 5 * 6: ]
ytest = y[33 * 5 * 6: ]

clf = XGBRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, early_stopping_round=50)
clf.fit(Xtrain, ytrain)
predictions = clf.predict(Xtest)

#Calculate MAE
MAE = mean_absolute_error(ytest, predictions)

#Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current prevalence, 1 otherwise.
increase = np.where(df.iloc[33 * 5 * 6: ]['next_prevalence'] < df.iloc[33 * 5 * 6: ]['prevalence'], 0, 1)
predicted_increase = np.where(predictions < df.iloc[33 * 5 * 6: ]['prevalence'], 0, 1)

#Calculate accuracy of predicted boolean increase/decrease
acc = accuracy_score(increase, predicted_increase)

#Print model parameters
print('no. of trees: ' + str(best_model_trees) + '\nmax_depth: ' + str(best_model_depth) + '\ncolumns: ' + str(important_feature[: best_model_columns_nr]))

#Print model scores
print(MAE, acc)
