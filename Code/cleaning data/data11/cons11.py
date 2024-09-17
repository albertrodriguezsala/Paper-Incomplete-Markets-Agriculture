# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:20:02 2018

@author: Albert
"""


# =============================================================================
# Consumption aggregation 2011-12
# =============================================================================

'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2011-12 (ISA-LSMS) and computes:
         - food consumption prices at different regional levels. To use to value production.
         - Consumption dataset at the household level.
Main outcome: cons11.csv
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

my_dirct = str(dirct)+'/data/raw data/2011/'
folder =  str(dirct)+'/data/data11/'
folder2 = str(dirct)+'/data/auxiliary data/'

dollars = 2586.89

#Import basic information (to construct regional prices)
basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False)
basic = basic[["HHID","region"]] 
basic.rename(columns={'HHID':'hh'}, inplace=True)



basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","region","urban",'h1aq1', "h1aq2", "h1aq3", "h1aq4","year", "month"]] 
basic.columns = ["hh","region","urban", 'district',"county", 'subcounty', 'parish', "year", "month"]
basic.rename(columns={'HHID':'hh'}, inplace=True)
basic['district'] = basic['district'].str.upper()
#basic['hh'] = pd.to_numeric(basic['hh'])

# FOOD CONSUMPTION-----------------------------------------------------------
c2 = pd.read_stata(my_dirct+'GSEC15B.dta', convert_categoricals=False)
c2.columns = ["hh","code","cons_cat","food_group","cons_7d","days", "unit","purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value","total_quant","total_value", "m_p", "gate_p"]


## convert to kg
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['kgconverter_11','kgconverter_med','unit','code','cropID']]

kg_units2 = pd.read_csv(folder2+'kg conversion/c_directkg_units.csv')
kg_units2 = kg_units2[['unit','kgconverter_direct']]

c2 = c2.merge(kg_units, on=['unit','code'], how='left')
c2 = c2.merge(kg_units2, on=['unit'], how='left')

c2['to_kg'] = c2['kgconverter_11'].fillna(c2['kgconverter_direct'])



quant_vars = ["purch_home","purch_away","own","gift", 'total']

for var in quant_vars:
    c2[[var+'_kg']] = c2[[var+"_quant"]]
    c2[var+'_kg'] = c2[var+"_quant"]*c2['to_kg']
    

c2['m_p'] =c2['purch_home_value']/c2['purch_home_kg']
c2['gate_p'] = c2['gate_p']/c2['to_kg']


## Get consumption food prices
pricescons = c2.groupby(by="code")[["m_p"]].median()
pricescons.to_csv(folder+"pricesfood11.csv")

## prices at regional level
c2 = c2.merge(basic, on='hh', how='right')
pricescons_reg = c2.groupby(by=["code", "region"])[["m_p"]].median()
pricescons_reg.to_csv(folder+"regionpricesfood11.csv")

## County level
pricescons_county = c2.groupby(by=["code", "county"])[["m_p"]].median()
pricescons_county.to_csv(folder+"countypricesfood11.csv")

pricescons_district = c2.groupby(by=["code", "district"])[["m_p"]].median()
pricescons_district.to_csv(folder+"districtpricesfood11.csv")

livestock = c2.loc[c2["code"].isin([117,118,119,120,121,122,123,124,125]),["hh","own_value"]]
livestock = livestock.groupby(by="hh").sum()*52

livestock.to_csv(folder+"c_animal11.csv")
suml = livestock.describe()/dollars

#Aggregate across items
c2 = c2.groupby(by="hh")[["purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value","total_quant","total_value"]].sum()
c2 = c2[["purch_home_value", "purch_away_value", "own_value","gift_value","total_value"]]

c2["cfood"] = c2[["purch_home_value", "purch_away_value", "own_value","gift_value"]].sum(axis=1)

c2.rename(columns={'gift_value':'cfood_gift'}, inplace=True)
c2.rename(columns={'own_value':'cfood_own'}, inplace=True)

c2["cfood_purch"] = c2.loc[:,["purch_home_value","purch_away_value"]].sum(axis=1)
c2["cfood_nogift"] = c2.loc[:,["cfood_purch","cfood_own"]].sum(axis=1)

# Food consumption at year level
c2 = c2[["cfood", "cfood_nogift", "cfood_own", "cfood_purch", "cfood_gift"]]*52
# Cfood is total value. cfood_nogift is total value minus gifts.
c2.reset_index(inplace=True)

data = c2



# NONFOOD NONDURABLE CONSUMPTION---------------------------------
c3 = pd.read_stata(my_dirct+'GSEC15C.dta', convert_categoricals=False)
c3.columns = ["hh","code","cons_30d","unit","purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value", "m_p"]

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


# DURABLE CONSUMPTION---------------------------------------------
c4 = pd.read_stata(my_dirct+'GSEC15D.dta', convert_categoricals=False)
c4.columns = ["hh","code","cons_30d","unit","purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value","m_p","total_q","total_value","result"]

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


cdata_short.to_csv(folder+"cons11.csv", index=False)