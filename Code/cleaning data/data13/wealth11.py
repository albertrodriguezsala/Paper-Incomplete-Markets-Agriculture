# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:47:47 2019

@author: rodri
"""

# =============================================================================
### Household Wealth: Uganda 2011-12 
# =============================================================================

'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2011-12 (ISA-LSMS) and computes:
         - household assets values.
    Uploads the cleaned data from the agricultural part and sums livestock and farm capital wealth
    Uploads data on land value (created from the surveys in a separate .py)
Main outcome: wealth11.csv
'''

hq=0.995

# Import main libraries and functions
import pandas as pd
import os
import numpy as np
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers


if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

os.chdir(dirct)

my_dirct = str(dirct)+'/data/raw data/2011/'
folder =  str(dirct)+'/data/data11/'
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
liquid_w = hhq14a.loc[ (hhq14a['item']=='Other Buildings') | (hhq14a['item']=='Land') | (hhq14a['item']=='Jewelry and Watches'), ['hh','asset_value']]
liquid_w = (liquid_w.groupby(by='hh')[['asset_value']].sum()).replace(0,np.nan)
liquid_w.columns = ['liquid_asset']
liquid_w.reset_index(inplace=True)
liquid_w['hh'] = pd.to_numeric(liquid_w['hh'])
assets_data = assets_data.merge(liquid_w, on='hh', how='left')




### Land reported
'''
land = hhq14a.loc[(hhq14a['item']=='Land'),['hh','asset_value']]
land = (land.groupby(by='hh')[['asset_value']].sum()).replace(0,np.nan)
land.columns = ['land_asset']
land.reset_index(inplace=True)
land['hh'] = pd.to_numeric(land['hh'])
assets_data = assets_data.merge(land, on='hh', how='left')
'''

#### Checking the data: Are there outliers? =================================
sum_assets = assets_data.describe()/dollars


# =============================================================================
# land value
# =============================================================================

land_hat = pd.read_csv(folder2+'land_value11.csv')


# =============================================================================
# farm capital and livestock value
# =============================================================================

agrls = pd.read_csv(folder+'wealth_agrls11.csv')



# =============================================================================
# Total Wealth
# =============================================================================

wealth = pd.merge(assets_data, agrls, on='hh', how='outer')
wealth = pd.merge(wealth, land_hat, on='hh', how='outer')
wealth['liquid_w'] = wealth['liquid_asset'].fillna(0) + wealth['wealth_lvstk'].fillna(0)


sum_wealth = wealth.describe()/dollars
#Save data  ====================================================================
wealth.to_csv(folder+"wealth11.csv", index=False)
print(sum_wealth)