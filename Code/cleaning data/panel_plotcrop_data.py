# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:29:34 2018

@author: rodri
"""

# =============================================================================
# Panel data hh-plot-crop level
# =============================================================================

# appends dataset 5 waves at hh-plot-crop level



import pandas as pd
import numpy as np
import os
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers, data_stats
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

folder =  str(dirct)

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.6, 6.4)
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

dollars = 2586.89

import warnings
warnings.filterwarnings('ignore')

### All monetary variables controlled for inflation and in 2013 US$


# IMPORT DATA 5 WAVES and append them =================================
## first I need to correct change ID in 2013 wave and onwards
data15 = pd.read_csv(folder+'/data/data15/agric_data15.csv', dtype={'cropID':str})
data15['wave'] = 2015

data13 = pd.read_csv(folder+'/data/data13/agric_data13.csv')
data13['wave'] = 2013
data13_all = pd.read_csv(folder+'/data/data13/data13.csv')
data13_all = data13_all[['hh','HHID_old']]
data13_all.columns = ['HHID', 'hh_old']

data15_ids = pd.DataFrame(data15['HHID'].unique())
data15_ids.columns = ['HHID']

data13.rename(columns={'HHID':'ggg'}, inplace=True)
data13['HHID'] = data13["ggg"].str.slice(0, 6, 1) + data13["ggg"].str.slice(10, 12, 1)

data13_all.rename(columns={'HHID':'ggg'}, inplace=True)
data13_all['HHID'] = data13_all["ggg"].str.slice(0, 6, 1) + data13_all["ggg"].str.slice(10, 12, 1)
data_ids = data13_all[['HHID','hh_old']].merge(data15_ids, on='HHID', how='outer')
data_ids['HHID'] = data_ids['HHID']
pd.value_counts(data_ids['HHID'])
data_ids['hh_old'].fillna(data_ids['HHID'].str.slice(1,8,1), inplace=True)
data_ids.rename(columns={'hh_old':'hh'}, inplace=True)
pd.value_counts(data_ids['hh'])
data_ids['hh'] = pd.to_numeric(data_ids['hh'])
## Get id back to the data
data13 = data13.merge(data_ids, on='HHID', how='inner')
del data13['HHID']
data15 = data15.merge(data_ids, on='HHID', how='inner')
del data15['HHID']

data13.rename(columns={'hh':'HHID'}, inplace=True)
data15.rename(columns={'hh':'HHID'}, inplace=True)

data11 = pd.read_csv(folder+'/data/data11/agric_data11.csv')
data11['wave'] = 2011

data10 = pd.read_csv(folder+'/data/data10/agric_data10.csv')
data10['wave'] = 2010

data09 = pd.read_csv(folder+'/data/data09/agric_data09.csv')
data09['wave'] = 2009

(data09[['y','m','A','HHID']].groupby(by='HHID').sum()).describe()

data11['HHID'] = data11['HHID'].astype('int64')
#data13.replace(np.nan,0, inplace=True)
#data15.replace(np.nan,0, inplace=True)
data13['HHID']  = data13['HHID'].astype('int64')
data15['HHID']  = data15['HHID'].astype('int64')

'''
data15.info(verbose=False, memory_usage="deep")
data13.info(verbose=False, memory_usage="deep")
data11.info(verbose=False, memory_usage="deep")
data10.info(verbose=False, memory_usage="deep")
data09.info(verbose=False, memory_usage="deep")
'''

data09_hh = data09.groupby(by=['HHID']).sum()
data10_hh = data10.groupby(by=['HHID']).sum()
data11_hh = data11.groupby(by=['HHID']).sum()
data13_hh = data13.groupby(by=['HHID']).sum()
data15_hh = data15.groupby(by=['HHID']).sum()

merge = data09_hh.merge(data10_hh, on='HHID', how='outer')
#2057
merge = merge.merge(data11_hh, on='HHID', how='outer')
#1900
merge = merge.merge(data13_hh, on='HHID', how='outer')
#0
merge = merge.merge(data15_hh, on='HHID', how='outer')


# Append waves ---------------------
data = pd.concat(list(data13.align(data15)), ignore_index=True)
data = pd.concat(list(data.align(data11)), ignore_index=True)
data = pd.concat(list(data.align(data10)), ignore_index=True)
data = data.append(data09)




# now the dataset with the 5 waves is created.
# Let's create some necessary variables

data.reset_index(inplace=True)

# homogeneize text data type accross waves
data['cropID'] = data['cropID'].str.title()

# intermediates variable at plot level
data['m'] = (data[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)).replace(0,np.nan)



## Checks the data and summary statistics
sumdata = data.describe()
count_crops = pd.value_counts(data['cropID'])
data_hhs = data.groupby(by=['HHID','wave']).sum().replace(0,np.nan)
data_hhs['A'] = data_hhs['A']/2


## summary 1: share of production to selling, own-consumption. etc
data_hhs['sell_share'] = data_hhs['sell_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['cons_share'] = data_hhs['cons_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['store_share'] = data_hhs['stored_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['gift_share'] = data_hhs['gift_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['interm_share'] = (data_hhs['animal_kg'].fillna(0)+data_hhs['food_prod_kg'].fillna(0)+data_hhs['seeds_kg'].fillna(0))/data_hhs['total_kg']

data_hhs['sell_yes'] = 1*(data_hhs['sell_kg']>10)
data_hhs['cons_yes'] = 1*(data_hhs['cons_kg']>10)
data_hhs['store_yes'] = 1*(data_hhs['stored_kg']>10)
data_hhs['gift_yes'] = 1*(data_hhs['gift_kg']>10)



data_hhs[['y','A','total_kg','m']].describe(percentiles=percentiles)
data_hhs.reset_index(inplace=True)

count_hhs = data_hhs.groupby(by=['HHID']).count()

data_hhs_avgwave = data_hhs.groupby(by='HHID').mean()
sum_hhlevel = data_hhs_avgwave[['y', 'A', 'm', 'total_kg', 'total2_kg', 'total_value_p_sell_reg', 'total_value_p_c_reg', 'total2_value_p_sell_reg', 'total2_value_p_c_reg']].describe()
data.replace(['Pawpaw', 'Sunflower', 'Jack Fruit', 'Millet', 'Coffe All', 'Cofee All'], ['Paw Paw', 'Sun Flower', 'Jackfruit', 'Finger Millet', 'Coffee All','Coffee All'], inplace=True)


data['y_over_A'] = data['y'] / data['A'].replace(0,np.nan)
data['kg_over_A'] = data['total2_kg']  / data['A'].replace(0,np.nan)
variables = ['m', 'l', 'A', 'y', 'y_over_A']
for var in variables:
    data['ln'+var] =  np.log(data[var].dropna()+np.abs(np.min(data[var]))).replace(-np.inf, np.nan)


data['t'] = np.nan
data.loc[(data['wave']==2009) & (data['season']==2),'t'] = 1 
data.loc[(data['wave']==2009) & (data['season']==1),'t'] = 2 
data.loc[(data['wave']==2010) & (data['season']==2),'t'] = 3 
data.loc[(data['wave']==2010) & (data['season']==1),'t'] = 4 
data.loc[(data['wave']==2011) & (data['season']==2),'t'] = 5 
data.loc[(data['wave']==2011) & (data['season']==1),'t'] = 6 
data.loc[(data['wave']==2013) & (data['season']==2),'t'] = 7 
data.loc[(data['wave']==2013) & (data['season']==1),'t'] = 8 
data.loc[(data['wave']==2015) & (data['season']==2),'t'] = 9 
data.loc[(data['wave']==2015) & (data['season']==1),'t'] = 10 


sum_labor = data[['y','l','hhlabor','hired_labor','labor_payment']]
w_lab = (data['labor_payment']/data['hired_labor']).median()


# remove non-crops in harvest
data = data.loc[(data['cropID']!='Other') & (data['cropID']!='Natural Pastures') &  (data['cropID']!='Plantation Trees')
& (data['cropID']!='Others')& (data['cropID']!='Fallow')& (data['cropID']!='Other forest trees')& (data['cropID']!='Bush')]

data.to_csv(folder+'/data/panel/panel_plotcrop_data.csv', index=False)

### The average across plots was 1.16
count_cropregion = data.groupby(by=['cropID','wave', 'region'])[['y']].count()
count_cropregion.columns = ['count']
count_cropregion.reset_index(inplace=True)

