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

# calculate enterprise value = [Market cap(mkt_cap)
#                               +(debt in current liabilities(dlcq)
#                               +long term debt (dlttq)) 
#                               + market value preferred equity (pstkq*prccq)
#                               + noncontrolling interest (mibtq)] 
#                               - [cash and cash eqiv (chechy)]
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

# PROBLEM
# NOTE: i think i need to use groupby here to make sure i do not overlap stocks w/ the shift method
# calculate log(% change 1 yr) for each fundamental item that is not negative value
df_temp = df3.drop(['earnings_yld'], axis=1)
df_temp = df_temp / df_temp.shift(4)

# calculate log(% change 1 yr) for each fundamental item that is not negative value
df4 = np.log( 1 + df_temp)

# make first 4 entries of each group NaN so data doesnt leak between groups (different companies)
df4.loc[df4.groupby('tic').head(4).index, :] = np.NaN
df4.groupby('tic').nth((0, 1, 2, 3))

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

# create final df
df_final = pd.concat([df6, df_isnil], axis='columns')

# fill NaN values
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final = df_final.groupby('tic').fillna(method='ffill', limit=1)
df_final = df_final.groupby('tic').fillna(0)

X = df_final








# create target variables: outperformance over one year of meadian of all stocks
# pct_chg.loc[pct_chg.groupby('tic').head(10).index, :]


# find median pct change 
pct_chg = df2[['%_prc_chg_1yr']]
median_pct_change = pct_chg.groupby('date')['%_prc_chg_1yr'].median()
pct_chg['median_pct_chg'] = pct_chg.groupby('date')['%_prc_chg_1yr'].median()

pct_chg.head(10)
# pct_chg.loc[pct_chg.groupby('tic').head(7).index, :]
# pct_chg.loc['AAP', :].head(20)
# df2.loc['AAP', ['prccq', 'atq']].head(20)


# create dictionary with median yearly percernt change for each quarter
dic = median_pct_change.to_dict()

from itertools import islice
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

n_items = take(6, dic.items())

dates = df2.index.get_level_values('date')
dates = pd.Series(dates)


med_prc_chg = dates.map(dic)

values = df2['%_prc_chg_1yr'].values

# create target data : 1 if outpreformed median % year change for that year, else 0 
y = (values > med_prc_chg)
y = y*1

# NOTE: since we want to predict return one year ahead, shift array 4 qts
y = y.shift(-4)
y = y.fillna(0)

# convert data to numpy ndarrays
y = y.as_matrix()
X = X.as_matrix()



















# RNN construction 

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import keras packages
import keras
from keras.layers import Dense
from keras.models import Sequential

# Instantiate MLP : model_1
n_cols = X.shape[1]
model_1 = Sequential()
model_1.add(Dense(200, activation='relu', kernel_initializer='uniform', input_shape=(n_cols,)))
model_1.add(Dense(200, activation='relu', kernel_initializer='uniform'))
model_1.add(Dense(100, activation='relu', kernel_initializer='uniform'))
model_1.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

# Import EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

# Compile the model
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# Train the model
model_1_training = model_1.fit(X_train, y_train, epochs=10, callbacks=[early_stopping_monitor], validation_data=(X_test, y_test))

# Create performance plot
import seaborn as sns
sns.set()
plt.plot(model_1_training.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

score, acc = model_1.evaluate(X_test, y_test)
print("score {0}, accuracy {1}".format(score, acc))



stocks = pd.DataFrame(['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS'])
stocks.to_csv('s_and_p.txt', index=None, sep=' ')
