# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:20:42 2018

@author: Albert
"""

# =============================================================================
# Consumption aggregation 2013
# =============================================================================

# =============================================================================
# Consumption aggregation 2009-10
# =============================================================================

'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2013 (ISA-LSMS) and computes:
         - food consumption prices at different regional levels. To use to value production.
         - Consumption dataset at the household level.
Main outcome: cons13.csv
'''
import pandas as pd
import numpy as np
import os
from statsmodels.formula.api import ols
from pathlib import Path


dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

os.chdir(dirct)

my_dirct = str(dirct)+'/data/raw data/2013/'
folder =  str(dirct)+'/data/data13/'
folder2 = str(dirct)+'/data/auxiliary data/'
# To pass all monetary variables to US 2013 $
dollars = 2586.89

#Import basic information (to construct regional prices)
basic = pd.read_csv(my_dirct+'gsec1.csv', header=0, na_values='NA')
basic = basic[["HHID",'HHID_old',"region","urban","year", "month","sregion"]] 
basic.rename(columns={'HHID':'hh'}, inplace=True)


basic = pd.read_csv(my_dirct+'gsec1.csv', header=0, na_values='NA')
basic = basic[["HHID",'HHID_old',"region","urban","year", "month","sregion", 'h1aq1a',  'h1aq3b', 'h1aq4b']] 
basic.columns = ["hh",'HHID_old',"region","urban","year", "month","sregion", 'district_code', 'subcounty', 'parish']
district_data = pd.read_csv(folder2+'district_codename.csv')
basic = basic.merge(district_data, on='district_code')
basic['subcounty'] = basic['subcounty'].str.upper()
### I lose 1200 obs with merging with subcounty 2011
county = pd.read_csv(folder2+'county_subcounty.csv')
basic = basic.merge(county, on='subcounty', how='left')



#% FOOD CONSUMPTION----------------------------------------
c2 = pd.read_csv(my_dirct+'gsec15b.csv', header=0, na_values='NA')
c2.columns = ["hh","code","food_group","cons_7d","cons_days","unit","purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value","total_quant","total_value", "m_p2", "gate_p","wgt_X"]

## convert to kg
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['kgconverter_13','kgconverter_med','unit','code','cropID']]

kg_units2 = pd.read_csv(folder2+'kg conversion/c_directkg_units.csv')
kg_units2 = kg_units2[['unit','kgconverter_direct']]

c2 = c2.merge(kg_units, on=['unit','code'], how='left')
c2 = c2.merge(kg_units2, on=['unit'], how='left')

c2['to_kg'] = c2['kgconverter_13'].fillna(c2['kgconverter_direct'])

quant_vars = ["purch_home","purch_away","own","gift", 'total']

for var in quant_vars:
    c2[[var+'_kg']] = c2[[var+"_quant"]]
    c2[var+'_kg'] = c2[var+"_quant"]*c2['to_kg']
    

c2['m_p'] =c2['purch_home_value']/c2['purch_home_kg']
c2['m_p2'] =c2['m_p2']/c2['purch_home_kg']
#c2['m_p_away'] =c2['purch_away_value']/c2['purch_away_kg']  # very few
c2['m_p'] = c2['m_p'].fillna(c2['m_p2'])
c2['gate_p'] = c2['gate_p']/c2['to_kg']

## Get consumption food prices
pricescons = c2.groupby(by=["code"])[["m_p"]].median()
pricescons.to_csv(folder+"pricesfood13.csv")

## prices at regional level
## prices at regional level
c2 = c2.merge(basic, on='hh', how='right')
pricescons_reg = c2.groupby(by=["code", "region"])[["m_p"]].median()
pricescons_reg.to_csv(folder+"regionpricesfood13.csv")

pricescons_county = c2.groupby(by=["code", "county"])[["m_p"]].median()
pricescons_county.to_csv(folder+"countypricesfood13.csv")

pricescons_district = c2.groupby(by=["code", "district"])[["m_p"]].median()
pricescons_district.to_csv(folder+"districtpricesfood13.csv")

## Get livestock own produced consumption: We need it to value livestock income
livestock = c2.loc[c2["code"].isin([117,118,119,120,121,122,123,124,125]),["hh","own_value"]]
livestock = livestock.groupby(by="hh").sum()*52
livestock.to_csv(folder+"c_animal13.csv")
sum_ls = livestock.describe()/dollars

#Aggregate across items: aggregate at hh level
c2 = c2.groupby(by="hh")[["purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value","total_quant","total_value"]].sum()
c2 = c2[["purch_home_value", "purch_away_value", "own_value","gift_value","total_value"]]
c2["cfood"] = c2[["purch_home_value", "purch_away_value", "own_value","gift_value"]].sum(axis=1)


c2.rename(columns={'gift_value':'cfood_gift'}, inplace=True)
c2.rename(columns={'own_value':'cfood_own'}, inplace=True)

sumfood = c2.describe()/dollars

c2["cfood_purch"] = c2.loc[:,["purch_home_value","purch_away_value"]].sum(axis=1)
c2["cfood_nogift"] = c2.loc[:,["cfood_purch","cfood_own"]].sum(axis=1)
# Food consumption at year level
c2 = c2[["cfood", "cfood_nogift", "cfood_own", "cfood_purch", "cfood_gift"]]*52

# Cfood is total value. cfood_nogift is total value minus gifts.

## Summary the data
sumfood = c2.describe()/dollars
c2.reset_index(inplace=True)


data = c2


# NONFOOD NONDURABLE CONSUMPTION-------------------------------------
c3 = pd.read_csv(my_dirct+'gsec15c.csv', header=0, na_values='NA')
c3.columns = ["hh","code","cons_30d","unit","purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value", "m_p", "wgt_X"]

#Aggregate across items
c3 = c3.groupby(by="hh")[["purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value"]].sum()
c3['cnodur'] = c3.fillna(0)["purch_value"] + c3.fillna(0)["own_value"] + c3.fillna(0)["gift_value"]
c3["cnodur_nogift"] = c3.loc[:,["purch_value","own_value"]].sum(axis=1)
c3.rename(columns={'gift_value':'cnodur_gift'}, inplace=True)
c3.rename(columns={'own_value':'cnodur_own'}, inplace=True)
c3.rename(columns={'purch_value':'cnodur_purch'}, inplace=True)

# non food non durable consumption at year level
c3 = c3[["cnodur", "cnodur_nogift", "cnodur_own", "cnodur_purch", "cnodur_gift"]]*12
c3.reset_index(inplace=True)

data = data.merge(c3, on="hh", how="outer")


# DURABLE CONSUMPTION--------------------------------------
c4 = pd.read_csv(my_dirct+'gsec15d.csv', header=0, na_values='NA')
c4.columns = ["hh","code","cons_y","purch_value","own_value","gift_value", "wgt_X"]

c4 = c4.groupby(by="hh")[["purch_value","own_value","gift_value"]].sum()

c4['cdur'] = c4.fillna(0)["purch_value"] + c4.fillna(0)["own_value"] + c4.fillna(0)["gift_value"]
c4["cdur_nogift"] = c4.loc[:,["purch_value","own_value"]].sum(axis=1)
c4.rename(columns={'gift_value':'cdur_gift'}, inplace=True)
c4.rename(columns={'own_value':'cdur_own'}, inplace=True)
c4.rename(columns={'purch_value':'cdur_purch'}, inplace=True)

#  durable consumption at year level
c4 = c4[["cdur", "cdur_nogift", "cdur_own", "cdur_purch", "cdur_gift"]]
c4.reset_index(inplace=True)

#%% JOIN DATA
data = data.merge(c4, on="hh", how="outer")
cdata_short = data[["hh","cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]]

sumc = cdata_short.describe()/dollars
print(sumc)
cdata_short.to_csv(folder+"cons13.csv", index=False)






