# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:06:24 2018

@author: Albert
"""

# =============================================================================
#  DATA 2013-14 WAVE
# =============================================================================

'''
DESCRIPTION
    -  Merge the previously cleaned datasets on agriculture, consumption, income, wealth, labor and business income, sociodemographic characteristics.
Also adds basic information variables from the household survey (as country, region, urban, etc)
   - Computes the consumption, income, and wealth at the household level.
   - deflates monetary variables with the CPI index from the worldbank (entire country) and converts them to US 2013 dollars.
   - trims the consumption, income at wealth for extreme outliers. trimming level from 2.5 to 0.5 depending on the variable.
   - Provides summary statistics of consumption, income, and wealth for the wave 2010.
Output: data10.csv (entire country) and data10_rural.csv (only rural)
'''

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.2f}'.format
import os
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers, data_stats

if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

os.chdir(dirct)

my_dirct = str(dirct)+'/data/raw data/2013/'
folder =  str(dirct)+'/data/data13/'
folder2 = str(dirct)+'/data/auxiliary data/'
import warnings
warnings.filterwarnings('ignore')

# To pass all monetary variables to US 2013 $
dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

#IMPORT DATA

basic = pd.read_csv(my_dirct+'gsec1.csv', header=0, na_values='NA')
basic = basic[["HHID",'HHID_old',"region","urban","year", "month","sregion", 'h1aq1a',  'h1aq3b', 'h1aq4b']] 
basic.columns = ["hh",'HHID_old',"region","urban","year", "month","sregion", 'district_code', 'subcounty', 'parish']
district_data = pd.read_csv(folder2+'district_codename.csv')
basic['subcounty'] = basic['subcounty'].str.upper()
basic = basic.merge(district_data, on='district_code')
### I lose 1200 obs with merging with subcounty 2011
county = pd.read_csv(folder2+'county_subcounty.csv')
basic = basic.merge(county, on='subcounty', how='left')

count_district = basic.groupby(by='district').count()
count_county = basic.groupby(by='county').count()
count_subcounty = basic.groupby(by='subcounty').count()
count_parish =  basic.groupby(by='parish').count()
count_months = basic.groupby(by=['year','month']).count() 



socio13 = pd.read_csv(folder+"sociodem13.csv")
socio13.drop(socio13.columns[0], axis=1, inplace= True)

basic = basic.merge(socio13, on="hh", how="left")
    
#Create some variables ====================================

basic["wave"] = "2013-2014"
basic["age_sq"] = basic.age**2

#Create dummies
dummies = pd.get_dummies(basic['region'])
dummies.columns = ["region1","region2","region3","region4"]
dummies.drop(["region1"], axis=1, inplace=True)
# 1:central, 2:Eastern, 3:Northern, 4:Western
basic = basic.join(dummies)
dummies = pd.get_dummies(basic['sex'])
dummies.columns = ["male","female"]
dummies.drop(["male"], axis=1, inplace=True)
basic = basic.join(dummies)

basic_rural = basic.loc[basic['urban']==0]

#%% Consumption
hq=0.99
lq=0.00
cons = pd.read_csv(folder+"cons13.csv")
# ctotal: food + nofood
# ctotal dur: food + nofood + durables
# ctotal gift: food + nofood of gifts
data = basic.merge(cons, on="hh", how="left")

data_rural = basic_rural.merge(cons, on='hh', how='left')

# Nationwide ---------
#Remove outliers
#data[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]] =remove_outliers(data[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]], hq=hq)

data["ctotal"] = data.loc[:,["cfood","cnodur"]].sum(axis=1)
data['ctotal'].replace(0,np.nan, inplace=True)

data[['ctotal','cfood','cnodur']].describe()

pd.value_counts(data['ctotal'])  # 355 households with 0 consumption...
pd.value_counts(data['cfood']) 

# Only positive obs
data["ctotal_dur"] = data.loc[:,["cfood","cnodur","cdur"]].sum(axis=1)

data["ctotal_gift"] = data.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)
data["ctotal_dur_gift"] = data.loc[:,["ctotal_gift","cdur_gift"]].sum(axis=1)

data["ctotal_own"] = data.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)
data["ctotal_dur_own"] = data.loc[:,["ctotal_own","cdur_own"]].sum(axis=1)


# Rural -----------
#data_rural[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]] =remove_outliers(data_rural[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]], hq=hq)
data_rural["ctotal"] = data_rural.loc[:,["cfood","cnodur"]].sum(axis=1)
data_rural['ctotal'].replace(0,np.nan, inplace=True)

data_rural['ctotal'].describe()

data_rural["ctotal_dur"] = data_rural.loc[:,["cfood","cnodur","cdur"]].sum(axis=1)

### Let's include durables in total consumption
#data_rural['ctotal'] = data_rural['ctotal_dur']

data_rural["ctotal_gift"] = data_rural.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)
data_rural["ctotal_dur_gift"] = data_rural.loc[:,["ctotal_gift","cdur_gift"]].sum(axis=1)

data_rural["ctotal_own"] = data_rural.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)
data_rural["ctotal_dur_own"] = data_rural.loc[:,["ctotal_own","cdur_own"]].sum(axis=1)

sumc = data_rural[['ctotal', 'cfood', 'cfood_gift', 'ctotal_dur', 'cdur']].describe()/dollars




#%% +Wealth
wealth = pd.read_csv(folder+'wealth13.csv')
data = data.merge(wealth, on='hh', how='left')
data_rural = data_rural.merge(wealth, on='hh', how='left')

# Nationwide
# remove outliers
hq=0.99
#data[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital', ]] = remove_outliers(data[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital', ]], hq=hq)
data['wtotal'] = data.loc[:,["asset_value", 'wealth_agrls', 'land_value_hat']].sum(axis=1)

# Rural 
hq=0.99
#data_rural[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital', ]] = remove_outliers(data_rural[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital',  ]], hq=hq)
data_rural['wtotal'] = data_rural.loc[:,["asset_value", 'wealth_agrls', 'land_value_hat']].sum(axis=1)
sum_wealth = data_rural[['wtotal',"asset_value", 'wealth_agrls','land_value_hat']].describe()/dollars



#%% Income: 
#labor & business income
lab_inc = pd.read_csv(folder+'income_hhsec13.csv', header=0, na_values='nan')
lab_inc = lab_inc[['hh','wage_total', 'bs_profit', 'bs_revenue', 'other_inc']]

#Agricultural income: 
ag_inc = pd.read_csv(folder+'inc_agsec13.csv', header=0, na_values='nan')
inc = pd.merge(lab_inc, ag_inc, on="hh", how="outer")
priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 

data = data.merge(inc, on='hh', how='left')
data_rural = data_rural.merge(inc, on='hh', how='left')


data[['other_inc','stored_value_p_c_district']] = remove_outliers(data[['other_inc','stored_value_p_c_district']],  hq=0.995)
data['liquid_w'] = data[['liquid_w','other_inc','stored_value_p_c_district']].sum(axis=1)

#data_rural[['other_inc','stored_value_p_c_district']] = remove_outliers(data_rural[['other_inc','stored_value_p_c_district']],  hq=0.995)
data_rural['liquid_w'] = data_rural[['asset_value','wealth_lvstk','other_inc','stored_value_p_c_district']].sum(axis=1)
data_rural['liquid_w'].replace([0,np.nan],inplace=True)


# Remove outliers income ==========================================
# Nationwide
hq=0.99
data['other_inc'] = remove_outliers(data[['other_inc']],  hq=hq)

data["profit_lvstk"] = remove_outliers(data[["profit_lvstk"]], lq=0.025, hq=0.975)
data["profit_lvstk"] = data["profit_lvstk"].replace(0,np.nan)

outliers = data.loc[(data['revenue_agr_p_c_district']>np.nanpercentile(data['revenue_agr_p_c_district'], 100)) |
                (data["wage_total"]>np.nanpercentile(data["wage_total"],98)) | 
                (data['bs_profit']>np.nanpercentile(data['bs_profit'],98)) |
                (data['bs_profit']<np.nanpercentile(data['bs_profit'],2.5)),'hh']

data = data[~data['hh'].isin(outliers)]
#rural ----------
hq=0.99

#other
data_rural['other_inc'] = remove_outliers(data_rural[['other_inc']], hq=0.975)
#livestock
data_rural["profit_lvstk"] = remove_outliers(data_rural[["profit_lvstk"]], lq=0.05, hq=0.99)
data_rural["profit_lvstk"] = data_rural["profit_lvstk"].replace(0,np.nan)


outliers = data_rural.loc[(data_rural['revenue_agr_p_c_district']>np.nanpercentile(data_rural['revenue_agr_p_c_district'],100)) |
                (data_rural["wage_total"]>np.nanpercentile(data_rural["wage_total"],98)) | 
                (data_rural['bs_profit']>np.nanpercentile(data_rural['bs_profit'],98)) |
                (data_rural['bs_profit']<np.nanpercentile(data_rural['bs_profit'],2.5)),'hh']

data_rural = data_rural[~data_rural['hh'].isin(outliers)]


# Get total income =======================
for p in priceslist:
    data["income_"+p] = data.loc[:,["wage_total","bs_profit","profit_lvstk", "revenue_agr_"+p, ]].sum(axis=1)   # 'other_inc','rent_owner','rent_noowner'
    data["revenue_"+p] = data.loc[:,["wage_total","bs_revenue","revenue_lvstk", "revenue_agr_"+p, ]].sum(axis=1) # 'other_inc','rent_owner','rent_noowner'
    data_rural["income_"+p] = data_rural.loc[:,["wage_total","bs_profit","profit_lvstk", "revenue_agr_"+p]].sum(axis=1)   
    data_rural["revenue_"+p] = data_rural.loc[:,["wage_total","bs_revenue","revenue_lvstk", "revenue_agr_"+p]].sum(axis=1)


sum_inc = data_rural[["income_p_sell_nat", "income_p_c_nat","income_p_sell_reg", "income_p_c_reg","income_p_sell_district", "income_p_c_district"]].describe()/dollars


data['inctotal'] = data['income_p_c_district']
data['inctotal'].replace(0, np.nan, inplace=True)
#inc.loc[inc['inctotal']<0, 'inctotal'] = 0

data_rural['inctotal'] = data_rural['income_p_c_district']
data_rural['inctotal'].replace(0, np.nan, inplace=True)


#%% Desinflate and convert to 2013 US$

# Substract for inflation and convert to US dollars
list_monvars = ['ctotal', 'cfood',"cnodur",'cdur','cfood_gift',  'ctotal_dur', 'ctotal_gift', 'asset_value', 'wealth_agrls','liquid_w', 'wealth_lvstk','land_value_hat', 'other_inc',
                'farm_capital', 'wtotal','inctotal',  'org_fert','chem_fert','pesticides','rent_owner','rent_noowner',
               "revenue_p_sell_nat", "revenue_p_c_nat","revenue_p_sell_reg", "revenue_p_c_reg","revenue_p_sell_county", "revenue_p_c_county","revenue_p_sell_district", "revenue_p_c_district",
               "revenue_agr_p_c_nat", "revenue_agr_p_c_reg", "revenue_agr_p_c_county","revenue_agr_p_c_district",
               "wage_total","bs_profit","profit_lvstk",
                 "income_p_sell_nat", "income_p_c_nat","income_p_sell_reg", "income_p_c_reg","income_p_sell_county", "income_p_c_county","income_p_sell_district", "income_p_c_district"]

for monetary_var in list_monvars:
    #data[[monetary_var]] = data[[monetary_var]].div(data.inflation, axis=0)
    data[[monetary_var]] = data[[monetary_var]]/dollars
    data_rural[[monetary_var]] = data_rural[[monetary_var]]/dollars
        
        



#%% Summarise data 


outliers2 = data.loc[(data['ctotal']<np.nanpercentile(data['ctotal'], 1)) | (data['ctotal']>np.nanpercentile(data['ctotal'], 99)) | 
                 (data["inctotal"]<np.nanpercentile(data["inctotal"],1)) | (data['inctotal']>np.nanpercentile(data['inctotal'], 99.5)) |          
                (data['wtotal']<np.nanpercentile(data['wtotal'], 1)) | (data['wtotal']>np.nanpercentile(data['wtotal'],100)),'hh']
data = data[~data['hh'].isin(outliers2)]

outliers2 = data_rural.loc[(data_rural['ctotal']<np.nanpercentile(data_rural['ctotal'], 1)) | (data_rural['ctotal']>np.nanpercentile(data_rural['ctotal'], 99)) | 
                 (data_rural["inctotal"]<np.nanpercentile(data_rural["inctotal"],1)) | (data_rural['inctotal']>np.nanpercentile(data_rural['inctotal'], 99.5)) |          
                (data_rural['wtotal']<np.nanpercentile(data_rural['wtotal'], 1)) | (data_rural['wtotal']>np.nanpercentile(data_rural['wtotal'],100)),'hh']

data_rural = data_rural[~data_rural['hh'].isin(outliers2)]


data.dropna(subset=['ctotal','inctotal'], inplace=True)
data_rural.dropna(subset=['ctotal','inctotal'], inplace=True)

### Generate per capita variables---------------------------
data[['ctotal_cap','inctotal_cap','wtotal_cap']] = data[['ctotal','inctotal','wtotal']].div(data.familysize, axis=0)

data_rural[['ctotal_cap','inctotal_cap','wtotal_cap']] = data_rural[['ctotal','inctotal','wtotal']].div(data_rural.familysize, axis=0)


#Summary Aggregates:
sumdata_hh = data_stats(data[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])
sumdata_hh


sumrural_hh = data_stats(data_rural[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])
sumrural_hh


sumdata_inc = data[['inctotal',"wage_total","bs_profit","profit_lvstk","revenue_agr_p_c_district", "revenue_agr_p_c_reg", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()


sumdata_inc_rural13 = data_rural[['inctotal',"wage_total","bs_profit","profit_lvstk","revenue_agr_p_c_district", "revenue_agr_p_c_reg", 'other_inc','rent_owner','rent_noowner']].describe()
data_rural2 = data.loc[data['urban']==0]
sumdata_inc_rural13_2 = data_rural2[['inctotal',"wage_total","bs_profit","profit_lvstk","revenue_agr_p_c_district", "revenue_agr_p_c_reg", 'other_inc','rent_owner','rent_noowner']].describe()



for item in ['ctotal', 'ctotal_dur', 'ctotal_gift', 'cfood', 'inctotal','wtotal', 'ctotal_cap','inctotal_cap','wtotal_cap']:
    data["ln"+item] = (np.log(data[item]+np.abs(np.min(data[item]))).replace([-np.inf, np.inf], np.nan)).dropna()
    #data["ln"+item] = np.log(data[item])
    
    data.rename(columns={"lnctotal":"lnc"}, inplace=True)
    data.rename(columns={"lninctotal":"lny"}, inplace=True)
    
    
    data_rural["ln"+item] = (np.log(data_rural[item]+np.abs(np.min(data_rural[item]))).replace([-np.inf, np.inf], np.nan)).dropna()
    #data["ln"+item] = np.log(data[item])
    
    data_rural.rename(columns={"lnctotal":"lnc"}, inplace=True)
    data_rural.rename(columns={"lninctotal":"lny"}, inplace=True)
    

print(data_stats(data_rural[['ctotal','inctotal','wtotal','liquid_w']]))

#Save Data
data.to_csv(folder+"data13.csv", index=False)
data_rural.to_csv(folder+"data13_rural.csv", index=False)


