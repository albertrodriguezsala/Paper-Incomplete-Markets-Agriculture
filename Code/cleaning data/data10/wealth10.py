9# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:47:48 2019

@author: rodri
"""

# =============================================================================
### Household Wealth: Uganda 2010-11
# =============================================================================


'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2010-11 (ISA-LSMS) and computes:
         - household assets values.
    Uploads the cleaned data from the agricultural part and sums livestock and farm capital wealth
    Uploads data on land value (created from the surveys in a separate .py)
Main outcome: wealth10.csv
'''

# Import main libraries and functions
import pandas as pd
import os
import numpy as np
from pathlib import Path
import sys
dirct  = Path('Master_data.py').resolve().parent
sys.path.append(str(dirct))

from data_functions_albert import remove_outliers

if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)


my_dirct = str(dirct)+'/data/raw data/2010/'
folder =  str(dirct)+'/data/data10/'
folder2 = str(dirct)+'/data/auxiliary data/'
# To pass all monetary variables to US 2013 $
dollars = 2586.89


# Import data =================================================================

hhq14a = pd.read_stata(my_dirct+'GSEC14.dta')
hhq14a = hhq14a[["HHID","h14q2","h14q3","h14q5"]]
hhq14a.columns = ["hh","item","asset_yes","asset_value"]
## Sum across items: aggregate at hh level
hhq14a = hhq14a[hhq14a['item']!='Land']
assets_data = hhq14a.groupby(by='hh')[['asset_value']].sum()  ##Need to write double [[]] to get it as dataframe. 
assets_data.reset_index(inplace=True)
assets_data['hh'] = pd.to_numeric(assets_data['hh'])


### get liquid wealth
### get liquid wealth
liquid_w = hhq14a.loc[ (hhq14a['item']=='Other Buildings') | (hhq14a['item']=='Land') | (hhq14a['item']=='Jewelry and Watches'), ['hh','asset_value']]
liquid_w = (liquid_w.groupby(by='hh')[['asset_value']].sum()).replace(0,np.nan)
liquid_w.columns = ['liquid_asset']
liquid_w.reset_index(inplace=True)
liquid_w['hh'] = pd.to_numeric(liquid_w['hh'])
assets_data = assets_data.merge(liquid_w, on='hh', how='left')


#### Checking the data: Are there outliers? =================================
sum_assets = assets_data.describe()/dollars


# =============================================================================
# land value
# =============================================================================
land = pd.read_csv(folder2+'land_value10.csv')

# =============================================================================
# farm capital and livestock value
# =============================================================================
agrls = pd.read_csv(folder+'wealth_agrls10.csv')


# =============================================================================
# Total Wealth
# =============================================================================

wealth = pd.merge(assets_data, agrls, on='hh', how='outer')
wealth = pd.merge(wealth, land, on='hh', how='outer')
wealth['liquid_w'] = wealth['liquid_asset'].fillna(0) + wealth['wealth_lvstk'].fillna(0)
sum_wealth = wealth.describe()/dollars
print(sum_wealth)
#Save data  ====================================================================
wealth.to_csv(folder+"wealth10.csv", index=False)