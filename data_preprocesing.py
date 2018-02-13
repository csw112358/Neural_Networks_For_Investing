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

# define a helper function: adjust_dates
def adjust_dates(x) :
    """Function to allign dates to match for each quarter end"""
    if(x.month == 1 or x.month == 2 or x.month == 3) :
        x = x.replace(month=1)
    elif(x.month == 4 or x.month == 5 or x.month == 6):
        x = x.replace(month=4)
    elif(x.month == 7 or x.month == 8 or x.month == 9):
        x = x.replace(month=7)
    elif(x.month == 10 or x.month == 11 or x.month == 12):
        x = x.replace(month=10)
    return x

# adjust dates to match per quarter for ease of processing:
# jan-mar = 1, apr-jun = 4, jul-sep = 7, oct-dec = 10
# change days to 28
df1['date'] = df1['date'].map(lambda x: x.replace(day=28))
df1['date'] = df1['date'].map(adjust_dates)

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
df1['%_prc_chg_1qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=1)
df1['%_prc_chg_2qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=2)
df1['%_prc_chg_3qtr'] = df1.groupby('tic')['prccq'].pct_change(periods=3)
df1['%_prc_chg_1yr'] = df1.groupby('tic')['prccq'].pct_change(periods=4)
# check that it worked:
# print(df1.loc[(slice(None), '2009-03-31'), ['prccq', '% price change 1qtr']])

# calculatre relative momentum rank for company months based on perecentile price change: mom_rank_1qtr,..., mom_rank_3qtr
df1['mom_rank_1qtr_%'] = df1.groupby('date')['%_prc_chg_1qtr'].rank(pct=True)
df1['mom_rank_2qtr_%'] = df1.groupby('date')['%_prc_chg_2qtr'].rank(pct=True)
df1['mom_rank_3qtr_%'] = df1.groupby('date')['%_prc_chg_3qtr'].rank(pct=True)
# check that it worked:
# print(df1.loc[(slice(None), '2009-12-31'), ['% price change 1qtr', 'mom_rank_1qtr']])

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








# fill NaN values : fill forward w/ limit=1, then fill w/ 0 : df2
df2 = df1.groupby('tic').fillna(method='ffill', limit=1)
df2 = df2.fillna(0)

# NORMALIZE fundamental items (use Frobenius norm) : df3
df3 = df2.drop(['%_prc_chg_1qtr', '%_prc_chg_2qtr', '%_prc_chg_3qtr', '%_prc_chg_1yr', 
                'mom_rank_1qtr_%', 'mom_rank_2qtr_%', 'mom_rank_3qtr_%', 'bk_to_mkt_rank_pct', 
                'earnings_yld_rank_pct'], axis=1)

df3_norm = np.linalg.norm(df3)
df3_normalized = df3 / df3_norm

# rename columns of df3 
df3_colnames = list(df3)
df3_colnames = ['norm:'+name for name in df3_colnames]
df3_normalized.columns = df3_colnames

# drop items that take negative values: df_temp
df_temp = df3.drop(['earnings_yld'], axis=1)
df_temp = df_temp / df_temp.shift(4)

# calculate log(% change 1 yr) for each fundamental item that is not negative value
df4 = np.log( 1 + df_temp)

# rename df4 columns
df4_colnames = list(df4)
df4_colnames = [name+'_1yr_chg' for name in df4_colnames]
df4.columns = df4_colnames

# create df_5 = relative momentum and relative value features
df5 = df1.loc[:, ['mom_rank_1qtr_%', 'mom_rank_2qtr_%', 'mom_rank_3qtr_%', 
                    'bk_to_mkt_rank_pct', 'earnings_yld_rank_pct']]

# df 6 = rel_momentum/val_df (df5)                          5 cols
#       + normalized_fundamentals_df (df3_normalized)       28 cols
#       + yr_chg_fundamentas_df (df4)                       28 cols

# create concatenated data frame and null value df
df6 = pd.concat([df4, df5, df3_normalized], axis='columns')

# create null matrix
df_isnil = df6.isnull() * 1
df_isnil_colnames = list(df_isnil)
df_isnil_colnames = ['nil?:'+name for name in df_isnil_colnames]
df_isnil.columns = df_isnil_colnames

df_final = pd.concat([df6, df_isnil], axis='columns')


df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final = df_final.groupby('tic').fillna(method='ffill', limit=1)
df_final = df_final.groupby('tic').fillna(0)


df_final.head(10)


