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
df1.drop(['prchq', 'prclq', 'costat', 'prcraq', 'uaptq'], axis=1, inplace=True)

# NOTE for this dataset you used accounts payable-utility and not accounts payable and accrued liabilities
# so all these entries are null (column: uaptq) 





# CALCULATION OF PCT CHANGES OF PRICE, MOMENTUM FEATURES, and VALUATION FEATURES 

# calculate percent price change: 1qtr, 2qtr, 3qtr, 1yr
df1['% price change 1qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=1)
df1['% price change 2qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=2)
df1['% price change 3qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=3)
df1['% price change 1yr'] = df1.groupby('tic')['prccq'].pct_change(periods=4)
# check that it worked:
# print(df1.loc[(slice(None), '2009-03-31'), ['prccq', '% price change 1qtr']])

# calculatre relative momentum rank for company months based on perecentile price change: mom_rank_1qtr,..., mom_rank_3qtr
df1['mom_rank_1qtr'] = df1.groupby('date')['% price change 1qtr'].rank()
df1['mom_rank_2qtr'] = df1.groupby('date')['% price change 2qtr'].rank()
df1['mom_rank_3qtr'] = df1.groupby('date')['% price change 3qtr'].rank()
# check that it worked:
# print(df1.loc[(slice(None), '2009-12-31'), ['% price change 1qtr', 'mom_rank_1qtr']])
# problem: I am not sure all quarterly dates are the same in df1, some start Jan 1 others March 1
# solution: I could create a conditional loop that changes one date scheme into the other if I know all date schemes
# i.e. loop through and change march to january, June to April, etc... 

# calculate: shareholders equity = total asets (atq) - total liabilities (ltq)
df1['shareholders_equity'] = df1['atq'] - df1['ltq']

# calc: market capitalization = #shares outstanding * price per share
df1['mkt_cap'] = df1['cshoq'] * df1['prccq']
# calculate book-to-market = shareholders equity / market capitalization 
df1['bk_to_mkt'] = df1['shareholders_equity'] / df1['mkt_cap']

# calculate enterprise value = [Market cap(mkt_cap)+(debt in current liabilities(dlcq)+long term debt (dlttq)) + (market value preferred equity = pstkq*prccq)+noncontrolling interest (mibtq)] - [cash and cash eqiv (chechy)]
# NOTE first I fill mibtq (non controlling interest) NaN values w/ zero for calculation
df1['mibtq'] = df1['mibtq'].fillna(0)
df1['entrprs_val']= (df1['mkt_cap'] + df1['dlcq'] + df1['dlttq'] + (df1['pstkq']*df1['prccq']) + df1['mibtq']) - (df1['chechy'])

# calculate earnings yield = operating income(oiadpq) / enterprise value(entrprs_val)
df1['earnings_yld'] = df1['oiadpq'] / df1['entrprs_val']

# calculate relative percentile ranking of each companymonth of book-to-market : bk_to_mkt_rank_pct
df1['bk_to_mkt_rank_pct'] = df1.groupby('date')['bk_to_mkt'].rank(pct=True)

# calc relative pct rank of earnings_yld per companymonth : earnings_yld_rank_pct
df1['earnings_yld_rank_pct'] = df1.groupby('date')['earnings_yld'].rank(pct=True) 



# NORMALIZE fundamental items (use Frobenius norm)







df1.info()
















# # testing/fiddling code (delete at end)
# df2 = pd.DataFrame({'tic': ['A', 'AAL', 'AAP', 'AAPL'], 'data':[]}
# print(df2)
# # useful functions: df.rank(), df.pct_change(), df.shift()

