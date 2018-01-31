"""This file takes in a .csv file containing company fundamental data and derives the data frame containing the model inputs for a RNN"""

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

# read the company data into a pandas dataframe: df1

filename = 's_and_p_A_ABBV_data.csv'
df1 = pd.read_csv(filename)

# format the date column to datetime object
time_format = '%Y%m%d'
df1['datadate'] = pd.to_datetime(df1['datadate'], format=time_format)
df1.rename(columns={'datadate': 'date'}, inplace=True)


# reindex to a hierarchical index with date and ticker symbol (tic)
df1.set_index(['tic', 'date'], inplace=True)
df1.sort_index(inplace=True)

# drop columns with unnecessary data
df1 = df1.loc[:, 'actq':]
df1.drop(['prchq', 'prclq', 'costat'], axis=1, inplace=True)

# NOTE for this dataset you used accounts payable-utility and not accounts payable and accrued liabilities
# so all these entries are null (column: uaptq) 

