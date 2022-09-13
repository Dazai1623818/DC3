'''------------SECTION IMPORTS---------------------'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm



'''------------SECTION USER VARIABLES--------------'''
#Define the path to your datafolder below
your_datapath = 'Data/'

#Define search space for number of trees in random forest and depth of trees
num_trees_min = 64
num_trees_max = 128

depth_min = 2
depth_max = 7



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

#Function that returns every possible subset (except the empty set) of the input list l
def subsets (l):
    subset_list = []
    for i in range(len(l) + 1):
        for j in range(i):
            subset_list.append(l[j: i])
    return subset_list



'''------------SECTION DATAFRAME CREATION--------------'''
#Create the dataframe for all districts
df = (make_combined_df_semiyearly(your_datapath)
      .dropna()
      .sort_values('date')
      .reset_index(drop=True)
      .query("district != ['Burco', 'Saakow', 'Rab Dhuure', 'Baydhaba', 'Afmadow']")
     )
#Drop disctricts with less than 7 observations: 'Burco', 'Saakow', 'Rab Dhuure', 'Baydhaba', 'Afmadow'



'''------------SECTION RANDOM FOREST CROSS VALIDATION--------------'''
#WARNING: this process can take some time, since there are a lot of hyperparameters to investigate. The search space can be manually reduced to speed up the process.

#Create empty list to store model scores
parameter_scores = []

#Define target and explanatory variables
X = df.drop(columns = ['increase', 'increase_numeric', 'date', 'district', 'prevalence', 'next_prevalence']) #Note that these columns are dropped, the remaining columns are used as explanatory variables
y = df['next_prevalence'].values

for num_trees in tqdm(range(num_trees_min, num_trees_max)):
    
    for depth in tqdm(range(depth_min, depth_max)):
        
        #Investigate every subset of explanatory variables
        for features in tqdm(subsets(X.columns)):
        
            #First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
            Xtrain = X[:99][features].copy().values
            ytrain = y[:99]
            Xtest = X[99:132][features].copy().values
            ytest = y[99:132]

            #Create a RandomForestRegressor with the selected hyperparameters and random state 0.
            clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

            #Fit to the training data
            clf.fit(Xtrain, ytrain)

            #Make a prediction on the test data
            predictions = clf.predict(Xtest)

            #Calculate mean absolute error
            MAE1 = mean_absolute_error(ytest, predictions)


            #Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
            Xtrain = X[:132][features].copy().values
            ytrain = y[:132]
            Xtest = X[132:165][features].copy().values
            ytest = y[132:165]

            #Create a RandomForestRegressor with the selected hyperparameters and random state 0.
            clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

            #Fit to the training data
            clf.fit(Xtrain, ytrain)

            #Make a prediction on the test data
            predictions = clf.predict(Xtest)

            #Calculate mean absolute error
            MAE2 = mean_absolute_error(ytest, predictions)

            #Calculate the mean MAE over the two folds
            mean_MAE = (MAE1 + MAE2)/2

            #Store the mean MAE together with the used hyperparameters in list 
            parameter_scores.append((mean_MAE, num_trees, depth, features))

#Sort the models based on score and retrieve the hyperparameters of the best model
parameter_scores.sort(key=lambda x: x[0])
best_model_score = parameter_scores[0][0]
best_model_trees = parameter_scores[0][1]
best_model_depth = parameter_scores[0][2]
best_model_columns = list(parameter_scores[0][3])



'''------------SECTION FINAL EVALUATION--------------'''
X = df[best_model_columns].values
y = df['next_prevalence'].values

#If there is only one explanatory variable, the values need to be reshaped for the model
if len(best_model_columns) == 1:
	X = X.reshape(-1, 1)

#Peform evaluation on full data
Xtrain = X[:165]
ytrain = y[:165]
Xtest = X[165:]
ytest = y[165:]

clf = RandomForestRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, random_state=0)
clf.fit(Xtrain, ytrain)
predictions = clf.predict(Xtest)

#Calculate MAE
MAE = mean_absolute_error(ytest, predictions)

#Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current prevalence, 1 otherwise.
# increase           = [0 if x<y else 1 for x in df.iloc[165:]['next_prevalence'] for y in df.iloc[165:]['prevalence']] (Wrong)
# predicted_increase = [0 if x<y else 1 for x in predictions                      for y in df.iloc[165:]['prevalence']] (Wrong)
increase = np.where(df.iloc[165:]['next_prevalence'] < df.iloc[165:]['prevalence'], 0, 1)
predicted_increase = np.where(predictions < df.iloc[165:]['prevalence'], 0, 1)

#Calculate accuracy of predicted boolean increase/decrease
acc = accuracy_score(increase, predicted_increase)

#Print model parameters
print('no. of trees: ' + str(best_model_trees) + '\nmax_depth: ' + str(best_model_depth) + '\ncolumns: ' + str(best_model_columns))

#Print model scores
print(MAE, acc)
