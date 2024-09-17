# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:20:08 2021

@author: rodri
"""
# =============================================================================
#  Estimating the value of the farming land across households
# =============================================================================


'''
DESCRIPTION:
    In the UNPS, only in the 2009-10 households were asked to report their land value.
    From this information, I compute per acre land prices across a set of land characteristics (categorical variables)
    location, tenure system, water source, topography, quality of land, and usage.
    
    From these set large set of prices, then I merge all the waves and I compute per each wave the estimated value of land per each of the household
    according to the household-wave characteristics of the plot ( location, tenure system, water source, topography, quality of land, and usage) times
    land size. I adjust for inflation using CPI index from the world bank.
    
    The correlation with this imputation land value and the actual reported land value for wave 2009 is 0.85!!
    
    Output: land_value09.csv, land_value10.csv, land_value11.csv, land_value13.csv, land_value15.csv
    
'''


import pandas as pd
import numpy as np
import os
os.chdir('C:/Users/rodri/Dropbox/JMP/python/auxiliary codes')
from data_functions_albert import remove_outliers
from statsmodels.formula.api import ols

os.chdir('C:/Users/rodri/Dropbox/JMP/data/raw data/2009/')
pd.options.display.float_format = '{:,.2f}'.format


folder =  'C:/Users/rodri/Dropbox/JMP/data/'


dollars = 2586.89

hq=0.99
lq=0.00


percentiles = [ 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

### GET LAND PRICES FROM 2009-10 WAVE -------------------------------

### households location
basic = pd.read_stata('GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","region","urban", 'h1aq1', "h1aq2b", "h1aq3b", "h1aq4b"]] 
basic.columns = ["HHID","region","urban", 'district_code', "county", 'subcounty', 'parish']
district_data = pd.read_csv(folder+'auxiliary data/district_codename.csv')
basic = basic.merge(district_data, on='district_code')


# Current land holdings
land09 = pd.read_stata('AGSEC2A.dta', convert_categoricals=False)
land09 = land09[["HHID", "a2aq2", 'a2aq4' ,'a2aq5','a2aq6','a2aq7','a2aq8', 'a2aq10', 'a2aq11','a2aq13a','a2aq13b','a2aq18','a2aq20','a2aq21', 'a2aq25']]
land09.columns = ['HHID','pltid','size_gps','size_reported','distance','tenure','acquired','sell_value', 'rent_1year','usage_a','usage_b','quality','water_source','topography', 'certificate']

land09['ratio_valuerent'] = land09['sell_value'].replace(0,np.nan)/land09['rent_1year'].replace(0,np.nan)

# Let's check prices
land09['land_size'] = land09['size_gps'].fillna(land09['size_reported'])

land09['p_acre'] = land09['sell_value'].replace(0,np.nan)/land09['land_size'].replace(0,np.nan)
land09['p_renting_acre'] = land09['rent_1year'].replace(0,np.nan)/land09['land_size'].replace(0,np.nan)


sum_landvalue = land09[['sell_value','rent_1year','p_acre', 'p_renting_acre']].describe(percentiles=percentiles)/dollars
sum_landsize = land09[['land_size','ratio_valuerent']].describe(percentiles=percentiles)

land09.reset_index(inplace=True)
outliers = land09.loc[(land09['land_size']>np.nanpercentile(land09['land_size'], 99)) |
                    
                (land09["p_acre"]<np.nanpercentile(land09["p_acre"],5)) | (land09["p_acre"]>np.nanpercentile(land09["p_acre"],95)) | 
                (land09['ratio_valuerent']<np.nanpercentile(land09['ratio_valuerent'], 5)) | (land09['ratio_valuerent']>np.nanpercentile(land09['ratio_valuerent'], 95)) ,'index']

land09 = land09[~land09['index'].isin(outliers)]

sum_landvalue2 = land09[['sell_value','rent_1year','p_acre', 'p_renting_acre']].describe(percentiles=percentiles)/dollars
sum_landsize2 = land09[['land_size','ratio_valuerent']].describe(percentiles=percentiles)

land09 = land09.merge(basic, on='HHID', how='left')


# Look at land characteristics

# Eliminate "other" or empty answers

land09[['usage_a','usage_b']] = land09[['usage_a', 'usage_b']].replace(96, np.nan)
land09[['tenure','topography']] = land09[['tenure', 'topography']].replace(6, np.nan)
land09[['acquired']] = land09[['acquired']].replace([5,6], np.nan)
county_list = pd.value_counts(land09['county'])
district_list = pd.value_counts(land09['district'])
land09[['county']] = land09[['county']].replace('', np.nan)
land09['topography'].replace([3,4],3, inplace=True) #1- hill 2- plain, 3-slope, 5-valley

median_byusage = land09[['usage_b','p_acre','sell_value', 'land_size']].groupby(by='usage_b').median()
median_bycounty = land09[['county','p_acre','sell_value', 'land_size']].groupby(by='county').median()

cat_vars = ['county','distance', 'district','tenure', 'usage_a','usage_b', 'quality','water_source','topography',]

for var in cat_vars:
    fit = ols('p_acre ~ C('+var+')', data=land09).fit(cov_type='HC1')
    print(fit.summary())


### Check join R2 --------------------------------------
#adj-R2 0.179
fit1 = ols('p_acre ~ C(county) ', data=land09).fit(cov_type='HC1') 
fit1.summary()

#0.18.1
fit11 = ols('p_acre ~ C(county)+ C(quality) ', data=land09).fit(cov_type='HC1') 
fit11.summary()


#adj-R2 0.183
fit2 = ols('p_acre ~ C(county) +C(tenure) ', data=land09).fit(cov_type='HC1') 
fit2.summary()

# 0.184
fit3 = ols('p_acre ~ C(county) + C(water_source)', data=land09).fit(cov_type='HC1') 
fit3.summary()

#0.185
fit4 = ols('p_acre ~ C(county) +C(topography) + C(quality)', data=land09).fit(cov_type='HC1') 
fit4.summary()


# 21.4 (best option)
fit7 = ols('p_acre ~ C(county) +C(topography) + C(quality) +C(usage_b) ', data=land09).fit(cov_type='HC1') 
fit7.summary()

# 21.9
# water source doesnt help
fit7 = ols('p_acre ~ C(county) +C(quality) +C(usage_b) +C(tenure) +C(topography) +C(certificate) ', data=land09).fit(cov_type='HC1') 
fit7.summary()


prices09_1 = land09[['district','county','quality','usage_a','tenure','topography','certificate','p_acre']].groupby(by=['district','county','quality','usage_a','tenure','topography','certificate']).median()
prices09_1.columns = ['p_acre_hat9']

#county median price
prices09_2 = land09[['county','p_acre']].groupby(by=['county']).median()
prices09_2.columns = ['p_acre_hat9_2']

#District median price
prices09_3 = land09[['district','p_acre']].groupby(by=['district']).median()
prices09_3.columns = ['p_acre_hat9_3']

# country median price
median_price = np.nanmedian(land09['p_acre'])



land09 = land09.merge(prices09_1,on=['district','county','quality','usage_a','tenure','topography','certificate'],how='left')
print(land09[['p_acre_hat9']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land09['p_acre_hat9'] = (land09['p_acre_hat9'].fillna(land09.groupby(['county','quality','usage_b','tenure','topography'])['p_acre_hat9'].fillna(method='ffill')))
print(land09[['p_acre_hat9']].isna().sum())
land09['p_acre_hat9'] = (land09['p_acre_hat9'].fillna(land09.groupby(['county','quality','usage_b','tenure'])['p_acre_hat9'].fillna(method='ffill')))
print(land09[['p_acre_hat9']].isna().sum())
land09['p_acre_hat9'] = (land09['p_acre_hat9'].fillna(land09.groupby(['county','quality','usage_b'])['p_acre_hat9'].fillna(method='ffill')))
print(land09[['p_acre_hat9']].isna().sum())
land09['p_acre_hat9'] = (land09['p_acre_hat9'].fillna(land09.groupby(['county','quality','usage_b'])['p_acre_hat9'].fillna(method='ffill')))
print(land09[['p_acre_hat9']].isna().sum())
land09['p_acre_hat9'] = (land09['p_acre_hat9'].fillna(land09.groupby(['county','quality'])['p_acre_hat9'].fillna(method='ffill')))
print(land09[['p_acre_hat9']].isna().sum())


land09 = land09.merge(prices09_2,on=['county'],how='left')
land09['p_acre_hat9'] = land09['p_acre_hat9'].fillna(land09['p_acre_hat9_2'])
print(land09[['p_acre_hat9']].isna().sum())

land09 = land09.merge(prices09_3,on=['district'],how='left')
land09['p_acre_hat9'] = land09['p_acre_hat9'].fillna(land09['p_acre_hat9_3'])
print(land09[['p_acre_hat9']].isna().sum())

land09['p_acre_hat9'] = land09['p_acre_hat9'].fillna(median_price)

## compute estimated land value
land09['land_value_hat_p09'] = land09['land_size']*land09['p_acre_hat9']

## check how good predictions are
land09[['p_acre', 'p_acre_hat9']].corr()

land09[['sell_value','land_value_hat_p09']].corr()

sumland = land09[['sell_value','land_value_hat_p09']].describe(percentiles=percentiles)





#%% Land value 2010

land10 = pd.read_stata(folder+'raw data/2010/AGSEC2A.dta', convert_categoricals=False)
basic10 = pd.read_stata(folder+'raw data/2010/GSEC1.dta', convert_categoricals=False )

basic10 = basic10[["HHID","region","urban", 'h1aq1', "h1aq2b", "h1aq3b", "h1aq4b","year", "month"]] 
basic10.columns = ["HHID","region","urban", 'district' , "county", 'subcounty', 'parish', "year", "month"]

land10 = land10[["HHID", "prcid", 'a2aq4' ,'a2aq5','a2aq6','a2aq7','a2aq8', 'a2aq10', 'a2aq11','a2aq13a','a2aq13b','a2aq18','a2aq20','a2aq21', 'a2aq25']]
land10.columns = ['HHID','prcdid','size_gps','size_reported','distance','tenure','acquired','sell_value', 'rent_1year','usage_a','usage_b','quality','water_source','topography', 'certificate']

land10['ratio_valuerent'] = land10['sell_value'].replace(0,np.nan)/land10['rent_1year'].replace(0,np.nan)

# Let's check prices
land10['land_size'] = land10['size_gps'].fillna(land10['size_reported'])
land10['p_acre'] = land10['sell_value'].replace(0,np.nan)/land10['land_size'].replace(0,np.nan)
land10['p_renting_acre'] = land10['rent_1year'].replace(0,np.nan)/land10['land_size'].replace(0,np.nan)

sum_landvalue = land10[['sell_value','rent_1year','p_acre', 'p_renting_acre']].describe(percentiles=percentiles)/dollars
sum_landsize = land10[['land_size','ratio_valuerent']].describe(percentiles=percentiles)

land10.reset_index(inplace=True)
outliers = land10.loc[(land10['land_size']>np.nanpercentile(land10['land_size'], 99)) |
                    
                (land10["p_acre"]<np.nanpercentile(land10["p_acre"],5)) | (land10["p_acre"]>np.nanpercentile(land10["p_acre"],95)) | 
                (land10['ratio_valuerent']<np.nanpercentile(land10['ratio_valuerent'], 5)) | (land10['ratio_valuerent']>np.nanpercentile(land10['ratio_valuerent'], 95)) ,'index']

land10 = land10[~land10['index'].isin(outliers)]

sum_landvalue2 = land10[['sell_value','rent_1year','p_acre', 'p_renting_acre']].describe(percentiles=percentiles)/dollars
sum_landsize2 = land10[['land_size','ratio_valuerent']].describe(percentiles=percentiles)
land10 = land10.merge(basic, on='HHID', how='left')

# Look at land characteristics

# first let's make some categorical variables coarser ------------

# Eliminate "other" or empty answers

land10[['usage_a','usage_b']] = land10[['usage_a', 'usage_b']].replace(96, np.nan)
land10[['tenure','topography']] = land10[['tenure','topography']].replace(6, np.nan)
land10[['acquired']] = land10[['acquired']].replace([5,6], np.nan)
county_list = pd.value_counts(land10['county'])
district_list = pd.value_counts(land10['district'])
land10[['county']] = land10[['county']].replace('', np.nan)

#land10['usage_a'].replace([1,2,3,4],1, inplace=True) #1, cultivated annual, 2- cultivated perenial, 3-rented-out, 4- cultivated by maillo.
land10['topography'].replace([3,4],3, inplace=True) #1- hill 2- plain, 3-slope, 5-valley


cat_vars = ['county','district','distance', 'tenure', 'usage_a','quality','water_source','topography',]

for var in cat_vars:
    fit = ols('p_acre ~ C('+var+')', data=land10).fit(cov_type='HC1')
    print(fit.summary())

### Let's comment a bit the results so that we leant something:
# (Note: All this relationships contain a lot of spurious correlation). 
# topography: flat lands are associated to lower prices. Hilly valleys are the ones that seem more expensive. Valley, slopes, not diff 0.
# distance: further away plots are significantly less valued
# tenure: Customary land much less value. 
# Usage: own cultivated the only ones that have significant impact
# soil type: significant impact. very similar btw fair and poor though
# water source: rainfed, swamp, increase price over irrigated. Weird, must be spurious.


### Check join R2 --------------------------------------
#adj-R2 0.19.6

fit1 = ols('p_acre ~ C(county) ', data=land10).fit(cov_type='HC1') 
fit1.summary()

#0.20.1
fit11 = ols('p_acre ~ C(county)+ C(quality) ', data=land10).fit(cov_type='HC1') 
fit11.summary()

# 20.6, distant variables still significant
fit2 = ols('p_acre ~ C(county) +C(quality) +C(distance) ', data=land10).fit(cov_type='HC1') 
fit2.summary()

#21.3
fit2 = ols('p_acre ~ C(county) +C(quality) +C(distance) +C(tenure) ', data=land10).fit(cov_type='HC1') 
fit2.summary()

# 21.9
fit3 = ols('p_acre ~ C(county) +C(quality) +C(distance) +C(tenure)  + C(water_source)', data=land10).fit(cov_type='HC1') 
fit3.summary()

#22.1
fit4 = ols('p_acre ~ C(county) +C(quality) +C(distance) +C(tenure)  + C(water_source) +C(topography)', data=land10).fit(cov_type='HC1') 
fit4.summary()

#23
fit7 = ols('p_acre ~C(county) +C(quality) +C(distance) +C(tenure)  + C(water_source) +C(topography) +C(usage_a)', data=land10).fit(cov_type='HC1') 
fit7.summary()

# R2 of 0.278. adjusted of 0.23. I choose to work with all the variables
fit7 = ols('p_acre ~ C(county) +C(quality) +C(distance) +C(tenure)  + C(water_source) +C(topography) +C(usage_a) +C(certificate)', data=land10).fit(cov_type='HC1') 
fit7.summary()

prices10_1 = land10[['county','quality','distance','usage_a','tenure','water_source','topography','p_acre']].groupby(by=['county','quality','distance','usage_a','tenure','water_source','topography']).median()
prices10_1.columns = ['p_acre_hat']

#county median price
prices10_2 = land10[['county','p_acre']].groupby(by=['county']).median()
prices10_2.columns = ['p_acre_hat_2']

#District median price
prices10_3 = land10[['district','p_acre']].groupby(by=['district']).median()
prices10_3.columns = ['p_acre_hat_3']

# country median price
median_price = np.nanmedian(land10['p_acre'])


### Merge prices with land parcels. First mant variables, then get coarser:
land10 = land10.merge(prices10_1,on=['county','quality','distance','usage_a','tenure','water_source','topography'],how='left')
print(land10[['p_acre_hat']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land10['p_acre_hat'] = (land10['p_acre_hat'].fillna(land10.groupby(['county','quality','distance','tenure','water_source','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land10[['p_acre_hat']].isna().sum())

land10['p_acre_hat'] = (land10['p_acre_hat'].fillna(land10.groupby(['county','quality','distance','tenure','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land10[['p_acre_hat']].isna().sum())
land10['p_acre_hat'] = (land10['p_acre_hat'].fillna(land10.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land10[['p_acre_hat']].isna().sum())
land10['p_acre_hat'] = (land10['p_acre_hat'].fillna(land10.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land10[['p_acre_hat']].isna().sum())
land10['p_acre_hat'] = (land10['p_acre_hat'].fillna(land10.groupby(['county','quality'])['p_acre_hat'].fillna(method='ffill')))
print(land10[['p_acre_hat']].isna().sum())

land10 = land10.merge(prices10_2,on=['county'],how='left')
land10['p_acre_hat'] = land10['p_acre_hat'].fillna(land10['p_acre_hat_2'])
print(land10[['p_acre_hat']].isna().sum())

land10 = land10.merge(prices10_3,on=['district'],how='left')
land10['p_acre_hat'] = land10['p_acre_hat'].fillna(land10['p_acre_hat_3'])
print(land10[['p_acre_hat']].isna().sum())

land10['p_acre_hat'] = land10['p_acre_hat'].fillna(median_price)

## compute estimated land value
land10['land_value_hat'] = land10['land_size']*land10['p_acre_hat']

## check how good predictions are
land10[['p_acre', 'p_acre_hat']].corr()
land10[['sell_value','land_value_hat']].corr()
sumland = land10[['sell_value','land_value_hat']].describe(percentiles=percentiles)
### 85% of correlation between reported land value and predicted using median prices!!!


land10.rename(columns={'HHID':'hh'}, inplace=True)

land10_short = land10[['hh', 'sell_value', 'land_value_hat']].groupby(by='hh').sum()

land10_short.to_csv(folder+'data10/land_value10.csv')



#%% check how good predictions are on 2009 dataset. -----------------------------

land09 = land09.merge(prices10_1,on=['county','quality','distance','usage_a','tenure','water_source','topography'],how='left')
print(land09[['p_acre_hat']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land09['p_acre_hat'] = (land09['p_acre_hat'].fillna(land09.groupby(['county','quality','distance','tenure','water_source','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land09[['p_acre_hat']].isna().sum())

land09['p_acre_hat'] = (land09['p_acre_hat'].fillna(land09.groupby(['county','quality','distance','tenure','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land09[['p_acre_hat']].isna().sum())
land09['p_acre_hat'] = (land09['p_acre_hat'].fillna(land09.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land09[['p_acre_hat']].isna().sum())
land09['p_acre_hat'] = (land09['p_acre_hat'].fillna(land09.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land09[['p_acre_hat']].isna().sum())
land09['p_acre_hat'] = (land09['p_acre_hat'].fillna(land09.groupby(['county','quality'])['p_acre_hat'].fillna(method='ffill')))
print(land09[['p_acre_hat']].isna().sum())


land09 = land09.merge(prices10_2,on=['county'],how='left')
land09['p_acre_hat'] = land09['p_acre_hat'].fillna(land09['p_acre_hat_2'])
print(land09[['p_acre_hat']].isna().sum())

land09 = land09.merge(prices10_3,on=['district'],how='left')
land09['p_acre_hat'] = land09['p_acre_hat'].fillna(land09['p_acre_hat_3'])
print(land09[['p_acre_hat']].isna().sum())

land09['p_acre_hat'] = land09['p_acre_hat'].fillna(median_price)

## compute estimated land value
inflation09 = 0.709890102/0.77845969
land09['land_value_hat_p10'] = land09['land_size']*land09['p_acre_hat']*inflation09

## check how good predictions are
land09[['p_acre', 'p_acre_hat9', 'p_acre_hat']].corr()
## 24%

land09[['sell_value','land_value_hat_p09','land_value_hat_p10']].corr()
## 51% ... Not that bad! though not that good.


land09.rename(columns={'HHID':'hh'}, inplace=True)

land09_short = land09[['hh', 'sell_value', 'land_value_hat_p09','land_value_hat_p10']].groupby(by='hh').sum()

land09_short.to_csv(folder+'data09/land_value09.csv')




#%% Land value 2011

## WORK ON HERE



basic11 = pd.read_csv('C:/Users/rodri/Dropbox/JMP/data/auxiliary data/basic11.csv')


inflation11 =  0.886005638/0.77845969

land11 = pd.read_stata(folder+'raw data/2011/AGSEC2A.dta', convert_categoricals=False )
land11 = land11[["HHID", "parcelID", 'a2aq4' ,'a2aq5','a2aq6','a2aq7','a2aq8', 'a2aq11a','a2aq11b','a2aq17','a2aq18','a2aq19', 'a2aq23']]
land11.columns = ['HHID','parcelID','size_gps','size_reported','distance','tenure','acquired','usage_a','usage_b','quality','water_source','topography', 'certificate']
#land11["HHID"] = pd.to_numeric(land11['HHID'], downcast='integer')

land11['land_size'] = land11['size_gps'].fillna(land11['size_reported'])

land11 = land11.merge(basic11, on='HHID', how='left')


# Eliminate "other" or empty answers
land11[['usage_a','usage_b']] = land11[['usage_a', 'usage_b']].replace(96, np.nan)
land11[['tenure','topography']] = land11[['tenure','topography']].replace(6, np.nan)
land11[['acquired']] = land11[['acquired']].replace([5,6], np.nan)
county_list = pd.value_counts(land11['county'])
district_list = pd.value_counts(land11['district'])
land11[['county']] = land11[['county']].replace('', np.nan)

land11['topography'].replace([3,4],3, inplace=True) #1- hill 2- plain, 3-slope, 5-valley



land11 = land11.merge(prices10_1,on=['county','quality','distance','usage_a','tenure','water_source','topography'],how='left')
print(land11[['p_acre_hat']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land11['p_acre_hat'] = (land11['p_acre_hat'].fillna(land11.groupby(['county','quality','distance','tenure','water_source','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land11[['p_acre_hat']].isna().sum())

land11['p_acre_hat'] = (land11['p_acre_hat'].fillna(land11.groupby(['county','quality','distance','tenure','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land11[['p_acre_hat']].isna().sum())
land11['p_acre_hat'] = (land11['p_acre_hat'].fillna(land11.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land11[['p_acre_hat']].isna().sum())
land11['p_acre_hat'] = (land11['p_acre_hat'].fillna(land11.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land11[['p_acre_hat']].isna().sum())
land11['p_acre_hat'] = (land11['p_acre_hat'].fillna(land11.groupby(['county','quality'])['p_acre_hat'].fillna(method='ffill')))
print(land11[['p_acre_hat']].isna().sum())

land11 = land11.merge(prices10_2,on=['county'],how='left')
land11['p_acre_hat'] = land11['p_acre_hat'].fillna(land11['p_acre_hat_2'])
print(land11[['p_acre_hat']].isna().sum())

land11[['district']] = land11['district'].str.title()
land11 = land11.merge(prices10_3,on=['district'],how='left')
land11['p_acre_hat'] = land11['p_acre_hat'].fillna(land11['p_acre_hat_3'])
print(land11[['p_acre_hat']].isna().sum())

land11['p_acre_hat'] = land11['p_acre_hat'].fillna(median_price)

## compute estimated land value
land11['land_value_hat'] = land11['land_size']*land11['p_acre_hat']*inflation11

land11.rename(columns={'HHID':'hh'}, inplace=True)

land11_short = land11[['hh', 'land_value_hat']].groupby(by='hh').sum()

land11_short.to_csv(folder+'data11/land_value11.csv')


#%% Land value 2013

basic13 = pd.read_csv(folder+'raw data/2013/gsec1.csv', header=0, na_values='NA')
basic13 = basic13[["HHID", "HHID_old","region","urban","year", "month","sregion", 'h1aq1a',  'h1aq3b', 'h1aq4b']] 
basic13.columns = ["hh", "hh_old","region","urban","year", "month","sregion", 'district_code', 'subcounty', 'parish']
district_data = pd.read_csv(folder+'auxiliary data/district_codename.csv')
basic13 = basic13.merge(district_data, on='district_code')
basic13['subcounty'] = basic13['subcounty'].str.upper()
### I lose 1200 obs with merging with subcounty 2011
county = pd.read_csv(folder+'auxiliary data/county_subcounty.csv')
basic13 = basic13.merge(county, on='subcounty', how='left')

inflation13 =  1/0.77845969

# Id problem
ag1 = pd.read_csv(folder+'raw data/2013/agsec1.csv', header=0, na_values='NA')
ag1= ag1[["hh","HHID"]] 

land13 = pd.read_csv(folder+'raw data/2013/agsec2a.csv', header=0, na_values='NA' )
land13 = land13[["HHID", "parcelID", 'a2aq4' ,'a2aq5','a2aq6','a2aq7','a2aq8', 'a2aq11a','a2aq11b','a2aq17','a2aq18','a2aq19', 'a2aq23']]
land13.columns = ['HHID','parcelID','size_gps','size_reported','distance','tenure','acquired','usage_a','usage_b','quality','water_source','topography', 'certificate']
#land13["HHID"] = pd.to_numeric(land13['HHID'], downcast='integer')

land13['land_size'] = land13['size_gps'].fillna(land13['size_reported'])

land13 = land13.merge(ag1, on='HHID', how='left')    #with inner loop I also don't lose observations
land13 = land13.merge(basic13, on='hh', how='left')

# Eliminate "other" or empty answers
land13[['usage_a','usage_b']] = land13[['usage_a', 'usage_b']].replace(96, np.nan)
land13[['tenure','topography']] = land13[['tenure','topography']].replace(6, np.nan)
land13[['acquired']] = land13[['acquired']].replace([5,6], np.nan)
county_list = pd.value_counts(land13['county'])
district_list = pd.value_counts(land13['district'])
land13[['county']] = land13[['county']].replace('', np.nan)

land13['topography'].replace([3,4],3, inplace=True) #1- hill 2- plain, 3-slope, 5-valley



land13 = land13.merge(prices10_1,on=['county','quality','distance','usage_a','tenure','water_source','topography'],how='left')
print(land13[['p_acre_hat']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land13['p_acre_hat'] = (land13['p_acre_hat'].fillna(land13.groupby(['county','quality','distance','tenure','water_source','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land13[['p_acre_hat']].isna().sum())

land13['p_acre_hat'] = (land13['p_acre_hat'].fillna(land13.groupby(['county','quality','distance','tenure','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land13[['p_acre_hat']].isna().sum())
land13['p_acre_hat'] = (land13['p_acre_hat'].fillna(land13.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land13[['p_acre_hat']].isna().sum())
land13['p_acre_hat'] = (land13['p_acre_hat'].fillna(land13.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land13[['p_acre_hat']].isna().sum())
land13['p_acre_hat'] = (land13['p_acre_hat'].fillna(land13.groupby(['county','quality'])['p_acre_hat'].fillna(method='ffill')))
print(land13[['p_acre_hat']].isna().sum())

land13 = land13.merge(prices10_2,on=['county'],how='left')
land13['p_acre_hat'] = land13['p_acre_hat'].fillna(land13['p_acre_hat_2'])
print(land13[['p_acre_hat']].isna().sum())

land13[['district']] = land13['district'].str.title()
land13 = land13.merge(prices10_3,on=['district'],how='left')
land13['p_acre_hat'] = land13['p_acre_hat'].fillna(land13['p_acre_hat_3'])
print(land13[['p_acre_hat']].isna().sum())

land13['p_acre_hat'] = land13['p_acre_hat'].fillna(median_price)

## compute estimated land value
land13['land_value_hat'] = land13['land_size']*land13['p_acre_hat']*inflation13


land13_short = land13[['hh', 'land_value_hat']].groupby(by='hh').sum()

land13_short.to_csv(folder+'data13/land_value13.csv', index=True)


#%% Land value 2015

basic15 = pd.read_csv(folder+'raw data/2015/gsec1.csv', header=0, na_values='NA')
basic15 = basic15[["HHID",'hh',"region","urban","year", "month","sregion",  'district', 'district_name', 'subcounty_name', 'parish_name']] 
basic15.columns = ["HHID",'hh_2',"region","urban","year", "month","sregion",  'district_code', 'district', 'subcounty', 'parish']
basic15['subcounty'] = basic15['subcounty'].str.upper()

### I lose 1300 obs with merging with subcounty 2011
county = pd.read_csv(folder+'auxiliary data/county_subcounty.csv')
basic15 = basic15.merge(county, on='subcounty', how='left')

inflation15 =  1.099176036/0.77845969

# Id problem
ag1 = pd.read_csv(folder+'raw data/2015/agsec1.csv', header=0, na_values='NA')
ag1= ag1[["hh",'hh_agric',"HHID"]]


land15 = pd.read_stata(folder+'raw data/2015/AGSEC2A.dta', convert_categoricals=False)
land15 = land15[["HHID", "parcelID", 'a2aq4' ,'a2aq5','a2aq6','a2aq7','a2aq8', 'a2aq11a','a2aq11b','a2aq17','a2aq18','a2aq19', 'a2aq23']]
land15.columns = ['HHID','parcelID','size_gps','size_reported','distance','tenure','acquired','usage_a','usage_b','quality','water_source','topography', 'certificate']
land15['land_size'] = land15['size_gps'].fillna(land15['size_reported'])

land15 = land15.merge(basic15, on='HHID', how='left')
land15 = land15.merge(ag1, on='HHID', how='left') #inner same number so it is good


# Eliminate "other" or empty answers
land15[['usage_a','usage_b']] = land15[['usage_a', 'usage_b']].replace(96, np.nan)
land15[['tenure','topography']] = land15[['tenure','topography']].replace(6, np.nan)
land15[['acquired']] = land15[['acquired']].replace([5,6], np.nan)
county_list = pd.value_counts(land15['county'])
district_list = pd.value_counts(land15['district'])
land15[['county']] = land15[['county']].replace('', np.nan)

land15['topography'].replace([3,4],3, inplace=True) #1- hill 2- plain, 3-slope, 5-valley



land15 = land15.merge(prices10_1,on=['county','quality','distance','usage_a','tenure','water_source','topography'],how='left')
print(land15[['p_acre_hat']].isna().sum())
### Fill missing nans by grouping median prices at coarser levels.
land15['p_acre_hat'] = (land15['p_acre_hat'].fillna(land15.groupby(['county','quality','distance','tenure','water_source','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land15[['p_acre_hat']].isna().sum())

land15['p_acre_hat'] = (land15['p_acre_hat'].fillna(land15.groupby(['county','quality','distance','tenure','topography'])['p_acre_hat'].fillna(method='ffill')))
print(land15[['p_acre_hat']].isna().sum())
land15['p_acre_hat'] = (land15['p_acre_hat'].fillna(land15.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land15[['p_acre_hat']].isna().sum())
land15['p_acre_hat'] = (land15['p_acre_hat'].fillna(land15.groupby(['county','quality','usage_a'])['p_acre_hat'].fillna(method='ffill')))
print(land15[['p_acre_hat']].isna().sum())
land15['p_acre_hat'] = (land15['p_acre_hat'].fillna(land15.groupby(['county','quality'])['p_acre_hat'].fillna(method='ffill')))
print(land15[['p_acre_hat']].isna().sum())

land15 = land15.merge(prices10_2,on=['county'],how='left')
land15['p_acre_hat'] = land15['p_acre_hat'].fillna(land15['p_acre_hat_2'])
print(land15[['p_acre_hat']].isna().sum())

land15[['district']] = land15['district'].str.title()
land15 = land15.merge(prices10_3,on=['district'],how='left')
land15['p_acre_hat'] = land15['p_acre_hat'].fillna(land15['p_acre_hat_3'])
print(land15[['p_acre_hat']].isna().sum())

land15['p_acre_hat'] = land15['p_acre_hat'].fillna(median_price)

## compute estimated land value
land15['land_value_hat'] = land15['land_size']*land15['p_acre_hat']*inflation15


land15_short = land15[['hh', 'land_value_hat']].groupby(by='hh').sum()

land15_short.to_csv(folder+'data15/land_value15.csv', index=True)




