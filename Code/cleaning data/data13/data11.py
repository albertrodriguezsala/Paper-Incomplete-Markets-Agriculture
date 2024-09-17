# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:19:32 2019

@author: rodri
"""


# =============================================================================
#  DATA 2011-12 WAVE
# =============================================================================

'''
DESCRIPTION
    -  Merge the previously cleaned datasets on agriculture, consumption, income, wealth, labor and business income, sociodemographic characteristics.
Also adds basic information variables from the household survey (as country, region, urban, etc)
   - Computes the consumption, income, and wealth at the household level.
   - deflates monetary variables with the CPI index from the worldbank (entire country) and converts them to US 2013 dollars.
   - trims the consumption, income at wealth for extreme outliers. trimming level from 2.5 to 0.5 depending on the variable.
   - Provides summary statistics of consumption, income, and wealth for the wave 2011.
Output: data11.csv (entire country) and data11_rural.csv (only rural)
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

my_dirct = str(dirct)+'/data/raw data/2011/'
folder =  str(dirct)+'/data/data11/'
folder2 = str(dirct)+'/data/auxiliary data/'
import warnings
warnings.filterwarnings('ignore')

# To pass all monetary variables to US 2013 $
dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

#IMPORT DATA

basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )

basic = basic[["HHID","region","urban",'h1aq1', "h1aq2", "h1aq3", "h1aq4","year", "month"]] 
basic.columns = ["hh","region","urban", 'district',"county", 'subcounty', 'parish', "year", "month"]


count_district = basic.groupby(by='district').count()
count_county = basic.groupby(by='county').count()
count_subcounty = basic.groupby(by='subcounty').count()
count_parish =  basic.groupby(by='parish').count()

basic.rename(columns={'HHID':'hh'}, inplace=True)
basic['hh'] = pd.to_numeric(basic['hh'])
count_months = basic.groupby(by=['year','month']).count() 



### world bank average btw 15 and 16 avg inflation
basic['inflation_avg'] = 0.886005638

socio11 = pd.read_csv(folder+"sociodem11.csv")
socio11.drop(socio11.columns[0], axis=1, inplace= True)

basic = basic.merge(socio11, on="hh", how="left")
    
#Create some variables ====================================
basic["wave"] = "2011-2012"
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
#trimming level
hq=0.99
lq=0.00

cons = pd.read_csv(folder+"cons11.csv")
# ctotal: food + nofood
# ctotal dur: food + nofood + durables
# ctotal gift: food + nofood of gifts
data = pd.merge(basic, cons, on="hh", how="left")

data_rural = pd.merge(basic_rural, cons, on='hh', how='left')

# Nationwide ---------
#Remove outliers
#data[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]] =remove_outliers(data[["cfood", "cnodur", "cdur", "cfood_gift","cnodur_gift", "cdur_gift", "cfood_nogift","cnodur_nogift","cdur_nogift","cfood_own","cnodur_own", "cdur_own"]], hq=hq)

data["ctotal"] = data.loc[:,["cfood","cnodur"]].sum(axis=1)
data['ctotal'].replace(0,np.nan, inplace=True)

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
data_rural["ctotal_dur"] = data_rural.loc[:,["cfood","cnodur","cdur"]].sum(axis=1)

### Let's include durables in total consumption
#data_rural['ctotal'] = data_rural['ctotal_dur']

data_rural["ctotal_gift"] = data_rural.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)
data_rural["ctotal_dur_gift"] = data_rural.loc[:,["ctotal_gift","cdur_gift"]].sum(axis=1)

data_rural["ctotal_own"] = data_rural.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)
data_rural["ctotal_dur_own"] = data_rural.loc[:,["ctotal_own","cdur_own"]].sum(axis=1)

sumc = data_rural[['ctotal', 'cfood', 'cfood_gift', 'ctotal_dur', 'cdur']].describe()


#%% +Wealth
wealth = pd.read_csv(folder+'wealth11.csv')
data = pd.merge(data, wealth, on='hh', how='left')
data_rural = pd.merge(data_rural, wealth, on='hh', how='left')

# Nationwide
# remove outliers
hq=0.99
#data[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital']] = remove_outliers(data[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital']], hq=hq)
data['wtotal'] = data.loc[:,["asset_value", 'wealth_agrls','land_value_hat']].sum(axis=1) 


# Rural 
hq=0.99
#data_rural[['asset_value', 'wealth_agrls','wealth_lvstk', 'farm_capital']] = remove_outliers(data_rural[['asset_value','wealth_agrls','wealth_lvstk', 'farm_capital']], hq=hq)
data_rural['wtotal'] = data_rural.loc[:,["asset_value", 'wealth_agrls','land_value_hat']].sum(axis=1) 
sum_wealth = data_rural[['wtotal',"asset_value", 'wealth_agrls','land_value_hat']].describe()/dollars




#%% Income: 
#labor & business income: 
lab_inc = pd.read_csv(folder+'income_hhsec_2011.csv', header=0, na_values='nan')
lab_inc = lab_inc[['hh','wage_total', 'bs_profit', 'bs_revenue', 'other_inc']]




#Agricultural income: 
ag_inc = pd.read_csv(folder+'inc_agsec11.csv', header=0, na_values='nan')
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


#livestock
data["profit_lvstk"] = remove_outliers(data[["profit_lvstk"]], lq=0.025, hq=0.975)
data["profit_lvstk"] = data["profit_lvstk"].replace(0,np.nan)

outliers = data.loc[(data['revenue_agr_p_c_district']>np.nanpercentile(data['revenue_agr_p_c_district'], 100)) |
                (data["wage_total"]>np.nanpercentile(data["wage_total"],97.5)) | 
                (data['bs_profit']>np.nanpercentile(data['bs_profit'],97.5)) |
                (data['bs_profit']<np.nanpercentile(data['bs_profit'],2.5)),'hh']

data = data[~data['hh'].isin(outliers)]

#rural ----------
hq=0.99

#other
data_rural['other_inc'] = remove_outliers(data_rural[['other_inc']],  hq=hq)
#livestock
data_rural["profit_lvstk"] = remove_outliers(data_rural[["profit_lvstk"]], lq=0.025, hq=0.975)
data_rural["profit_lvstk"] = data_rural["profit_lvstk"].replace(0,np.nan)

outliers = data_rural.loc[(data_rural['revenue_agr_p_c_district']>np.nanpercentile(data_rural['revenue_agr_p_c_district'], 100)) |
                (data_rural["wage_total"]>np.nanpercentile(data_rural["wage_total"],97.5)) | 
                (data_rural['bs_profit']>np.nanpercentile(data_rural['bs_profit'],97.5)) |
                (data_rural['bs_profit']<np.nanpercentile(data_rural['bs_profit'],2.5)),'hh']

data_rural = data_rural[~data_rural['hh'].isin(outliers)]



# Get total income =======================
for p in priceslist:
    data["income_"+p] = data.loc[:,["wage_total","bs_profit","profit_lvstk", "revenue_agr_"+p]].sum(axis=1)  #, 'other_inc','rent_owner','rent_noowner' 
    data["revenue_"+p] = data.loc[:,["wage_total","bs_revenue","revenue_lvstk", "revenue_agr_"+p]].sum(axis=1)
    data_rural["income_"+p] = data_rural.loc[:,["wage_total","bs_profit","profit_lvstk", "revenue_agr_"+p]].sum(axis=1)   
    data_rural["revenue_"+p] = data_rural.loc[:,["wage_total","bs_revenue","revenue_lvstk", "revenue_agr_"+p]].sum(axis=1)



sum_inc = data_rural[["income_p_sell_nat", "income_p_c_nat","income_p_sell_reg", "income_p_c_reg","income_p_sell_district", "income_p_c_district"]].describe()/dollars


data['inctotal'] = data['income_p_c_district']
data['inctotal'].replace(0, np.nan, inplace=True)
#inc.loc[inc['inctotal']<0, 'inctotal'] = 0

data_rural['inctotal'] = data_rural['income_p_c_district']
data_rural['inctotal'].replace(0, np.nan, inplace=True)
#inc.loc[inc['inctotal']<0, 'inctotal'] = 0


#%% Desinflate and convert to 2013 US$


# Substract for inflation and convert to US dollars
list_monvars = ['ctotal', 'cfood',"cnodur",'cdur', 'ctotal_dur', 'ctotal_gift','cfood_gift',  'asset_value', 'wealth_agrls','liquid_w', 'wealth_lvstk','land_value_hat','other_inc',
                'farm_capital', 'wtotal','inctotal',  'org_fert','chem_fert','pesticides','rent_owner','rent_noowner',
                "wage_total","bs_profit","profit_lvstk",  "revenue_agr_p_c_nat", "revenue_agr_p_c_reg", "revenue_agr_p_c_county","revenue_agr_p_c_district",
               "revenue_p_sell_nat", "revenue_p_c_nat","revenue_p_sell_reg", "revenue_p_c_reg","revenue_p_sell_county", "revenue_p_c_county","revenue_p_sell_district", "revenue_p_c_district",
                 "income_p_sell_nat", "income_p_c_nat","income_p_sell_reg", "income_p_c_reg","income_p_sell_county", "income_p_c_county","income_p_sell_district", "income_p_c_district"]

for monetary_var in list_monvars:
    data[[monetary_var]] = data[[monetary_var]].div(data.inflation_avg, axis=0)
    data[[monetary_var]] = data[[monetary_var]]/dollars
    data_rural[[monetary_var]] = data_rural[[monetary_var]].div(data_rural.inflation_avg, axis=0)
    data_rural[[monetary_var]] = data_rural[[monetary_var]]/dollars
        
        

#%% Summarise data 

# Remove outliers at CIW level
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
sumdata_inc_rural11 = data_rural[['inctotal',"wage_total","bs_profit","profit_lvstk", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()
data_rural2 = data.loc[data['urban']==0]
sumdata_inc_rural11_2 = data_rural2[['inctotal',"wage_total","bs_profit","profit_lvstk", "revenue_agr_p_c_nat", 'other_inc','rent_owner','rent_noowner']].describe()


print(data_stats(data_rural[['liquid_w']]))


### Generate per capita variables---------------------------
data[['ctotal_cap','inctotal_cap','wtotal_cap']] = data[['ctotal','inctotal','wtotal']].div(data.familysize, axis=0)

data_rural[['ctotal_cap','inctotal_cap','wtotal_cap']] = data_rural[['ctotal','inctotal','wtotal']].div(data_rural.familysize, axis=0)

sumdata_hh = data_stats(data[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])

sumrural_hh = data_stats(data_rural[["ctotal", "inctotal", 'wtotal','ctotal_cap','inctotal_cap','wtotal_cap']])



for item in ['ctotal', 'ctotal_dur', 'ctotal_gift', 'cfood', 'inctotal','wtotal', 'ctotal_cap','inctotal_cap','wtotal_cap']:
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


print(data_stats(data_rural[['ctotal','inctotal','wtotal','liquid_w']]))

#Save Data
data.to_csv(folder+"data11.csv", index=False)
data_rural.to_csv(folder+"data11_rural.csv", index=False)



