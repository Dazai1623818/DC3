README for Baseline.py

This document aims to give you context and explanation on how to use/set up and read Baseline.py. 
Background
	Zero Hunger Labs uses a random forest model to forecast GAM prevalence, you can find further informatiom in "Predictying Wasting_Data challenge background document.pdf"
	The file "Baseline.py" contains a similar model implementation, where the main difference from ZHL model is in cross validation and evaluation methods
	You can see the required packages and its versions listed on 'Requirements' at the end of this document

How to read Baeline.py

	The code is organized in six sections, namely:
		IMPORTS
		USER VARIABLE
		FUNCTIONS
		DATAFRAME CREATION
		RANDOM FOREST CROSS VALIDATION
		FINAL EVALUATION
	All sections have comments to guide you on understanding what the code does (input, purpose, output), you can identified the comments as they are precced by the character #
	There are 3 functions
		1) Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries
			def make_district_df_semiyearly(datapath, district_name)

		2) Function that combines the semiyearly dataset (from the function make_district_df_semiyearly) of all districts
			def make_combined_df_semiyearly(datapath)
			
		3) Function that returns every possible subset (except the empty set) of the input list l, It is used to investigate every subset of explanatory variables
			def subsets (l)
	The calculations for random forest, cross validation, and its evaluation are made using python function 
	The results of MAE and accuracy of the model are displayed as part of the section:FINAL EVALUATION,using print function
	
	
How to set up
	Be sure that the environment has the specifications described in section Requirements;
	Adjust the path to your datafolder, see SECTION USER VARIABLE;
	The function SECTION RANDOM FOREST CROSS VALIDATION can take processing time before it generates results, read the section before you run the code;
	
	
Example of a section
	All sections are written as comments using the feature '''----SECTION  [name]  -----''', an example of imports section is showed below:
	'''------------SECTION IMPORTS---------------------'''
	import pandas as pd
	import numpy as np
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_absolute_error, accuracy_score
	
Example of a function
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


Requirements
	matplotlib==3.5.1
	numpy==1.22.4
	pandas==1.3.3
	scikit_learn==1.1.2