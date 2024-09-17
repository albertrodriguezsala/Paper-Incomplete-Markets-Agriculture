# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:27:07 2019

@author: rodri
"""


# =============================================================================
#  DATA 2010-11 WAVE
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

my_dirct = str(dirct)+'/data/raw data/2010/'
folder =  str(dirct)+'/data/data10/'
folder2 = str(dirct)+'/data/auxiliary data/'

import warnings
warnings.filterwarnings('ignore')

# To pass all monetary variables to US 2013 $
dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

#%%IMPORT DATA

basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )

basic = basic[["HHID","region","urban", 'h1aq1', "h1aq2b", "h1aq3b", "h1aq4b","year", "month"]] 
basic.columns = ["hh","region","urban", 'district' , "county", 'subcounty', 'parish', "year", "month"]

count_district = basic.groupby(by='district').count()
count_county = basic.groupby(by='county').count()
count_subcounty = basic.groupby(by='subcounty').count()
count_parish =  basic.groupby(by='parish').count()



count_months = basic.groupby(by=['year','month']).count() 
basic['hh'] = pd.to_numeric(basic['hh'])
basic.dropna(subset=['year','month'] ,inplace=True)
basic.reset_index(inplace=True)
del basic['index']
'''
#Generate inflation variable (per individual)
inflation = pd.read_excel("inflation_UGA_09_14.xlsx", index_col=0)
lulu = []
for i in range(0,len(basic)):
    a = inflation.loc[basic.loc[i,"year"], basic.loc[i,"month"]]
    lulu.append(a)    
basic["inflation"] = pd.Series(lulu)

del inflation, i, lulu, a
'''
basic['inflation_avg'] = 0.77845969

socio10 = pd.read_csv(folder+"sociodem10.csv")
socio10.drop(socio10.columns[0], axis=1, inplace= True)

basic = basic.merge(socio10, on="hh", how="left")
    
#Create some variables ====================================
basic["wave"] = "2010-2011"
basic["age_sq"] = basic.age**2

## Region 0 is kampala. Make it central
basic['region'].replace(0,1,inplace=True)

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
#trimming level
hq=0.99
lq=0.00

cons = pd.read_csv(folder+"cons10.csv")
# ctotal: food + nofood
# ctotal dur: food + nofood + durables
# ctotal gift: food + nofood of gifts
data = pd.merge(basic, cons, on="hh", how="left")

data_rural = pd.merge(basic_rural, cons, on='hh', how='left')

# Nationwide ---------
#Remove outliers
#data[["cfood", "cnodur", "cfood_gift","cnodur_gift",  "cfood_nogift","cnodur_nogift","cfood_own","cnodur_own"]] =remove_outliers(data[["cfood", "cnodur", "cfood_gift","cnodur_gift",  "cfood_nogift","cnodur_nogift","cfood_own","cnodur_own"]], hq=hq)

data["ctotal"] = data.loc[:,["cfood","cnodur"]].sum(axis=1)
data['ctotal'].replace(0,np.nan, inplace=True)

data["ctotal_dur"] = data.loc[:,["ctotal","cdur"]].sum(axis=1)
data['ctotal_dur'].replace(0,np.nan, inplace=True)

data["ctotal_gift"] = data.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)
data["ctotal_own"] = data.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)


# Rural -----------
#data_rural[["cfood", "cnodur", "cfood_gift","cnodur_gift",  "cfood_nogift","cnodur_nogift","cfood_own","cnodur_own"]] =remove_outliers(data_rural[["cfood", "cnodur", "cfood_gift","cnodur_gift",  "cfood_nogift","cnodur_nogift","cfood_own","cnodur_own"]], hq=hq)
data_rural["ctotal"] = data_rural.loc[:,["cfood","cnodur"]].sum(axis=1)
data_rural['ctotal'].replace(0,np.nan, inplace=True)

data_rural["ctotal_dur"] = data_rural.loc[:,["ctotal","cdur"]].sum(axis=1)
data_rural['ctotal_dur'].replace(0,np.nan, inplace=True)

### Let's include durables in total consumption
#data_rural['ctotal'] = data_rural['ctotal_dur']

data_rural["ctotal_gift"] = data_rural.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)

data_rural["ctotal_own"] = data_rural.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)

sumc = data_rural[['ctotal', 'cfood', 'cfood_gift']].describe()





#%% +Wealth
wealth = pd.read_csv(folder+'wealth10.csv')

data = pd.merge(data, wealth, on='hh', how='left')
data_rural = pd.merge(data_rural, wealth, on='hh', how='left')

# Nationwide
# remove outliers
hq=0.99
#data[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital', 'liquid_w']] = remove_outliers(data[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital',  'liquid_w']], hq=hq)
data['wtotal'] = data.loc[:,["asset_value", 'wealth_agrls','land_value_hat']].sum(axis=1) #


# Rural 
hq=0.99
#data_rural[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital', 'liquid_w']] = remove_outliers(data_rural[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital',  'liquid_w']], hq=hq)
data_rural['wtotal'] = data_rural.loc[:,["asset_value", 'wealth_agrls','land_value_hat']].sum(axis=1)
sum_wealth = data_rural[['wtotal',"asset_value", 'wealth_agrls','land_value_hat']].describe()/dollars #,'land_value_hat'



 
#%% Income: 
#labor & business income: 
lab_inc = pd.read_csv(folder+'income_hhsec_2010.csv', header=0, na_values='nan')
lab_inc = lab_inc[['hh','wage_total', 'bs_profit', 'bs_revenue','other_inc']]

#Agricultural income: 
ag_inc = pd.read_csv(folder+'inc_agsec10.csv', header=0, na_values='nan')
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
data_rural['other_inc'] = remove_outliers(data_rural[['other_inc']],  hq=hq)
#livestock
data_rural["profit_lvstk"] = remove_outliers(data_rural[["profit_lvstk"]], lq=0.05, hq=0.98)
data_rural["profit_lvstk"] = data_rural["profit_lvstk"].replace(0,np.nan)
'''
#labor
data_rural[["wage_total"]] = remove_outliers(data_rural[["wage_total"]], hq=0.99)
#business
data_rural[['bs_profit','bs_revenue']] = remove_outliers(data_rural[['bs_profit','bs_revenue']], lq=0, hq=0.975)
data_rural['bs_profit'] = remove_outliers(data_rural[['bs_profit']], lq=0.05)

#agriculture
for p in priceslist:
    data_rural[['profit_agr_'+p,'revenue_agr_'+p]] = remove_outliers(data_rural[['profit_agr_'+p,'revenue_agr_'+p]], hq=0.975)
'''

## OPEN QUESTION: Should I delete the row or converted in as a nan (like right now)?

outliers = data_rural.loc[(data_rural['revenue_agr_p_c_district']>np.nanpercentile(data_rural['revenue_agr_p_c_district'], 100)) |
                (data_rural["wage_total"]>np.nanpercentile(data_rural["wage_total"],98)) | 
                (data_rural['bs_profit']>np.nanpercentile(data_rural['bs_profit'],98)) |
                (data_rural['bs_profit']<np.nanpercentile(data_rural['bs_profit'],2.5)),'hh']

data_rural = data_rural[~data_rural['hh'].isin(outliers)]



# Get total income =======================
for p in priceslist:
    data["income_"+p] = data.loc[:,["wage_total","bs_profit","profit_lvstk", "revenue_agr_"+p]].sum(axis=1) #, 'other_inc','rent_owner','rent_noowner'  
    data["revenue_"+p] = data.loc[:,["wage_total","bs_revenue","revenue_lvstk", "revenue_agr_"+p]].sum(axis=1)
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
list_monvars = ['ctotal', 'cfood',"cnodur",'cfood_gift', 'ctotal_gift', 'asset_value', 'wealth_agrls','liquid_w', 'wealth_lvstk','land_value_hat',
                'farm_capital', 'wtotal','inctotal',  'org_fert','chem_fert','pesticides','rent_owner','rent_noowner',
                "revenue_agr_p_c_nat", "revenue_agr_p_c_reg", "revenue_agr_p_c_county","revenue_agr_p_c_district",
                "wage_total","bs_profit","profit_lvstk",'other_inc',
               "revenue_p_sell_nat", "revenue_p_c_nat","revenue_p_sell_reg", "revenue_p_c_reg","revenue_p_sell_county", "revenue_p_c_county","revenue_p_sell_district", "revenue_p_c_district",
                 "income_p_sell_nat", "income_p_c_nat","income_p_sell_reg", "income_p_c_reg","income_p_sell_county", "income_p_c_county","income_p_sell_district", "income_p_c_district"]

for monetary_var in list_monvars:
    data[[monetary_var]] = data[[monetary_var]].div(data.inflation_avg, axis=0)
    data[[monetary_var]] = data[[monetary_var]]/dollars
    data_rural[[monetary_var]] = data_rural[[monetary_var]].div(data_rural.inflation_avg, axis=0)
    data_rural[[monetary_var]] = data_rural[[monetary_var]]/dollars
      
        



#%% Summarise data 


outliers2 = data.loc[(data['ctotal']<np.nanpercentile(data['ctotal'], 1)) | (data['ctotal']>np.nanpercentile(data['ctotal'], 99)) | 
                 (data["inctotal"]<np.nanpercentile(data["inctotal"],1)) | (data['inctotal']>np.nanpercentile(data['inctotal'], 100)) |          
                (data['wtotal']<np.nanpercentile(data['wtotal'], 1)) | (data['wtotal']>np.nanpercentile(data['wtotal'],100)),'hh']

data = data[~data['hh'].isin(outliers2)]

outliers2 = data_rural.loc[(data_rural['ctotal']<np.nanpercentile(data_rural['ctotal'], 1)) | (data_rural['ctotal']>np.nanpercentile(data_rural['ctotal'], 99)) | 
                 (data_rural["inctotal"]<np.nanpercentile(data_rural["inctotal"],1)) | (data_rural['inctotal']>np.nanpercentile(data_rural['inctotal'], 100)) |          
                (data_rural['wtotal']<np.nanpercentile(data_rural['wtotal'], 1)) | (data_rural['wtotal']>np.nanpercentile(data_rural['wtotal'],100)),'hh']

data_rural = data_rural[~data_rural['hh'].isin(outliers2)]


data.dropna(subset=['ctotal','inctotal'], inplace=True)
data_rural.dropna(subset=['ctotal','inctotal'], inplace=True)

sumdata_inc = data[['inctotal',"wage_total","bs_profit","profit_lvstk", "revenue_agr_p_c_reg", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()
sumdata_inc_rural10 = data_rural[['inctotal',"wage_total","bs_profit","profit_lvstk", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()
sum_wealth = data_rural[['wtotal',"asset_value", 'wealth_agrls','land_value_hat']].describe()



data_rural2 = data.loc[data['urban']==0]
sumdata_inc_rural10_2 = data_rural2[['inctotal',"wage_total","bs_profit","profit_lvstk", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()


### Generate per capita variables---------------------------
data[['ctotal_cap','inctotal_cap','wtotal_cap']] = data[['ctotal','inctotal','wtotal']].div(data.familysize, axis=0)
data_rural[['ctotal_cap','inctotal_cap','wtotal_cap']] = data_rural[['ctotal','inctotal','wtotal']].div(data_rural.familysize, axis=0)

sumdata_hh = data_stats(data[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])
sumrural_hh = data_stats(data_rural[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])



for item in ['ctotal', 'ctotal_gift', 'cfood', 'inctotal','wtotal', 'ctotal_cap','inctotal_cap','wtotal_cap']:
    data["ln"+item] = (np.log(data[item]+np.abs(np.min(data[item]))).replace([-np.inf, np.inf], np.nan)).dropna()
    #data["ln"+item] = np.log(data[item])
    
    data.rename(columns={"lnctotal":"lnc"}, inplace=True)
    data.rename(columns={"lninctotal":"lny"}, inplace=True)
    #data = data.drop_duplicates(subset=['hh'], keep=False)
    
    data_rural["ln"+item] = (np.log(data_rural[item]+np.abs(np.min(data_rural[item]))).replace([-np.inf, np.inf], np.nan)).dropna()
    #data["ln"+item] = np.log(data[item])
    
    data_rural.rename(columns={"lnctotal":"lnc"}, inplace=True)
    data_rural.rename(columns={"lninctotal":"lny"}, inplace=True)
    #data_rural = data_rural.drop_duplicates(subset=['hh'], keep=False)



print(data_rural[['ctotal','inctotal','wtotal','liquid_w']].describe())
#Save Data
data.to_csv(folder+"data10.csv", index=False)
data_rural.to_csv(folder+"data10_rural.csv", index=False)