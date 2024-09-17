# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:18:55 2018

@author: Albert
"""

# =============================================================================
#  Panel data UNPS (5 waves): 2009-10 to 2015-16 
# =============================================================================

'''
DESCRIPTION
        imports the datasets for each wave dataXX.csv and dataXX_rural.csv
        and outputs
        unbalanced panel all Uganda: panel_UGA.csv
        unbalanced panel rural Uganda: panel_rural_UGA.csv
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import  data_stats
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

folder =  str(dirct)+'/data/'

import warnings
warnings.filterwarnings('ignore')

# NATIONWIDE----------------
#Import 2009
data09 = pd.read_csv(folder+'data09/data09.csv')
#Import 2010
data10 = pd.read_csv(folder+'data10/data10.csv')
#Import 2011
data11 = pd.read_csv(folder+'data11/data11.csv')
#Import 2013
data13 = pd.read_csv(folder+'data13/data13.csv')
#Import 2015
data15 = pd.read_csv(folder+'data15/data15.csv')


dollars = 2586.89 

data13['HHID'] = data13["hh"].str.slice(0, 6, 1) + data13["hh"].str.slice(10, 12, 1)
del data13['hh'], data15['hh']
data_ids = data13[['HHID','HHID_old']].merge(data15[['HHID']], on='HHID', how='inner')

pd.value_counts(data_ids['HHID'])
data_ids['HHID_old'].fillna(data_ids['HHID'].str.slice(1,8,1), inplace=True)
data_ids.rename(columns={'HHID_old':'hh'}, inplace=True)
pd.value_counts(data_ids['hh'])
data_ids['hh'] = pd.to_numeric(data_ids['hh'])
## Get id back to the data
data13 = data13.merge(data_ids, on='HHID', how='left')
del data13['HHID']
data15 = data15.merge(data_ids, on='HHID', how='left')
del data15['HHID']
# Create panel



counthh_xwave = []
i=0
for data in [data09, data10, data11, data13, data15]:
    i+=1
    counthh = pd.value_counts(data['hh'])
    #print('wave'+str(i)+' count hh ='+str(max(counthh)))
    counthh_xwave.append(counthh)

data13 = data13.drop_duplicates(subset=['hh'],keep='first')
data15 = data15.drop_duplicates(subset=['hh'],keep='first')

panel = data09.append(data10)
panel = panel.append(data11)
panel = panel.append(data13)
panel = panel.append(data15)

panel['y_agric'] = panel['revenue_agr_p_c_district']


panel.to_csv(folder+"panel/panel_UGA.csv", index=False)
panel_0911 = panel.loc[(panel['wave']=='2009-2010') |(panel['wave']=='2010-2011') |(panel['wave']=='2011-2012') ]

# rural----------------

datarural09 = pd.read_csv(folder+'data09/data09_rural.csv')
sumrural_hh = datarural09[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']].describe()
datarural10 = pd.read_csv(folder+'data10/data10_rural.csv')
datarural11 = pd.read_csv(folder+'data11/data11_rural.csv')
datarural13 = pd.read_csv(folder+'data13/data13_rural.csv')
datarural15 = pd.read_csv(folder+'data15/data15_rural.csv')

sumrural_hh = data_stats(datarural15[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])


datarural13['HHID'] = datarural13["hh"].str.slice(0, 6, 1) + datarural13["hh"].str.slice(10, 12, 1)
del datarural13['hh'], datarural15['hh']
datarural_ids = datarural13[['HHID','HHID_old']].merge(datarural15[['HHID']], on='HHID', how='inner')

pd.value_counts(datarural_ids['HHID'])
datarural_ids['HHID_old'].fillna(datarural_ids['HHID'].str.slice(1,8,1), inplace=True)
datarural_ids.rename(columns={'HHID_old':'hh'}, inplace=True)
pd.value_counts(datarural_ids['hh'])
datarural_ids['hh'] = pd.to_numeric(datarural_ids['hh'])
## Get id back to the data
datarural13 = datarural13.merge(datarural_ids, on='HHID', how='left')
del datarural13['HHID']
datarural15 = datarural15.merge(datarural_ids, on='HHID', how='left')
del datarural15['HHID']
# Create panel
panel_rural = datarural09.append(datarural10)
panel_rural = panel_rural.append(datarural11)
panel_rural = panel_rural.append(datarural13)
panel_rural = panel_rural.append(datarural15)


panel_rural['y_agric'] = panel_rural['revenue_agr_p_c_district']


## Create balanced panel: 
counthh = panel_rural.groupby(by="hh")[["hh"]].count()
counthh.columns = ["counthh"]
counthh.reset_index(inplace=True)
panel = panel.merge(counthh, on="hh", how="left")


#Only those observed across 5 waves 
panelbal = panel.loc[panel["counthh"]==5,]
counthh = panelbal.groupby(by="hh")[["hh"]].count()


## Create balanced panel:
counthh = panel_rural.groupby(by="hh")[["hh"]].count()
counthh.columns = ["counthh"]
counthh.reset_index(inplace=True)
panel_rural = panel_rural.merge(counthh, on="hh", how="left")


#Only those observed across 5 waves 
panelbal_rural = panel_rural.loc[panel["counthh"]==5,]
counthh = panelbal_rural.groupby(by="hh")[["hh"]].count()


rural09 = panel_rural.loc[panel_rural['wave']=='2009-2010']

rural09[['ctotal','inctotal','wtotal','liquid_w']].describe()

panel_rural['liquid_w'].mean()

panel_rural_0911 = panel_rural.loc[(panel_rural['wave']=='2009-2010') |(panel_rural['wave']=='2010-2011') |(panel_rural['wave']=='2011-2012') ]

panel_rural['cdur'] = panel_rural['cdur']/dollars

data_urban = panel.loc[panel['urban']==1]


panel_rural.to_csv(folder+"panel/panel_rural_UGA.csv", index=False)

panel_rural['liquid_w'].describe()

