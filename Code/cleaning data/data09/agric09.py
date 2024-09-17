# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:12:51 2018

@author: rodri
"""

# =============================================================================
# #### Agricultural data Analysis Uganda 2009-10
# =============================================================================

'''
DESCRIPTION:
    Uses the data from the 2009-2010 Integrated Survey in Agriculture from the UNPS (ISA-LSMS) to obtain:
        For the 2 main harvest of the year of the survey. (fall 2009 and spring 2010/)
            -agricultural inputs variables at plot level and household level.
            -agricultural ouptut variables at plot level, household level, and crop level.   
        -household land, household farming assets (capital) household level
        - Livestock revenues (including non-sold consumption), costs, and stock (wealth) household level
     output: inc_agsec09.csv (agric income and costs at hh level) wealth_agrls09.csv (agric wealth: livestock+farming capital), agric_data09.csv (outputs and inputs at crop-plot-hh level)
'''

import pandas as pd
import numpy as np
import os
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


my_dirct = str(dirct)+'/data/raw data/2009/'  
folder =  str(dirct)+'/data/data09/'
folder2 = str(dirct)+'/data/auxiliary data/'

pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89


# Removing outliers
hq=0.99
lq=0.00

percentiles = [ 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

save=True


### households location
basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","region","urban", 'h1aq1', "h1aq2b", "h1aq3b", "h1aq4b"]] 
basic.columns = ["HHID","region","urban", 'district_code', "county", 'subcounty', 'parish']
district_data = pd.read_csv(folder2+'district_codename.csv')
basic = basic.merge(district_data, on='district_code')
blu = basic['district'].value_counts()

#%% AGRICULTURAL SEASON 1

    
#rent obtained------------------------------
ag2a = pd.read_stata(my_dirct+'AGSEC2A.dta')
ag2a = ag2a[["HHID", "a2aq16"]]
ag2a = ag2a.groupby(by="HHID")[["a2aq16"]].sum()
ag2a.columns = ["rent_owner"]
ag2a.reset_index(inplace=True)
ag2a["HHID"] = pd.to_numeric(ag2a['HHID'])

# rent payment--------------------------
ag2b = pd.read_stata(my_dirct+'AGSEC2B.dta')
ag2b = ag2b[["HHID", "a2bq9", "a2bq16"]]
ag2b = ag2b.groupby(by="HHID")[["a2bq9", "a2bq16"]].sum()

ag2b["rent_noowner"] = ag2b["a2bq16"].fillna(0) - ag2b["a2bq9"].fillna(0)
ag2b["rent_noowner"] = ag2b["rent_noowner"].replace(0, np.nan)
ag2b = ag2b[["rent_noowner"]]
ag2b.reset_index(inplace=True)
ag2b["HHID"] = pd.to_numeric(ag2b['HHID'])

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3a = pd.read_stata(my_dirct+'agsec3a.dta')
ag3a = ag3a[["HHID", 'a3aq3', 'a3aq5','a3aq7',"a3aq8", 'a3aq16',"a3aq18", "a3aq19",'a3aq28b', 'a3aq30', 'a3aq31', 'a3aq38', 'a3aq39', 'a3aq42a', 'a3aq42b','a3aq42c','a3aq43']]

ag3a['hhlabor'] = ag3a["a3aq39"].fillna(0) 
ag3a['hired_labor'] = ag3a["a3aq42a"].fillna(0)+ag3a["a3aq42b"].fillna(0)+ag3a["a3aq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3a['p_orgfert'] = np.nanmedian( ag3a['a3aq8'].div(ag3a['a3aq7'], axis=0) )
ag3a['p_chemfert'] = np.nanmedian( ag3a['a3aq19'].div(ag3a['a3aq18'], axis=0) )
ag3a['p_pest'] = np.nanmedian( ag3a['a3aq31'].div(ag3a['a3aq30'], axis=0) )  ## missing quantity pesticies purchased in the data (but in the questionnaire is there)

ag3a['org_fert'] = ag3a['p_orgfert']*ag3a['a3aq5']
### important changes when included non-bought 
ag3a['org_fert'].describe()
ag3a['a3aq8'].describe()

## with chemfert might not be important
ag3a['chem_fert'] = ag3a['p_chemfert']*ag3a['a3aq16']
ag3a['chem_fert'].describe()
ag3a['a3aq18'].describe()

##pesticides: small changes
ag3a['pesticides'] = ag3a['p_pest']*ag3a['a3aq28b']
ag3a['pesticides'].describe()
ag3a['a3aq31'].describe()

ag3a = ag3a[["HHID", 'a3aq3', "org_fert", "chem_fert", "pesticides",'hhlabor', 'hired_labor', "a3aq43"]]
ag3a.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4a = pd.read_stata(my_dirct+'agsec4a.dta')
ag4a = ag4a[["HHID", 'a4aq2','a4aq5','a4aq6' , 'a4aq8',  'a4aq9', "a4aq11"]]

ag4a.columns = ["HHID", 'pltid','cropID', 'crop_code', 'total_area', 'weightcrop', 'seed_cost']
ag4a['weightcrop'].replace(np.nan,100, inplace=True)

### already ask total area and proportion per crop
ag4a['area_planted'] = ag4a['total_area'].multiply(ag4a['weightcrop']/100, axis=0)


### For the variables that we don't know per crop within same plot (labor, intermediates). 
ag4a['weight'] = ag4a['weightcrop']

ag4a.reset_index(inplace=True)
ag4a.rename(columns={'index':'i'}, inplace=True)
agrica = pd.merge(ag3a, ag4a, on=['HHID','pltid'], how='right')
agrica.drop_duplicates(subset=['i'], inplace=True) 

### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agrica['weight']/100, axis=0)


# =============================================================================
# Output
# =============================================================================

ag5a = pd.read_stata(my_dirct+'agsec5a.dta')
ag5a = ag5a[["HHID",'a5aq3',"a5aq4","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq22"]]
ag5a.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]

ag5a['cropID'] = ag5a['cropID'].str.title()
ag5a.replace(['Avacoda', 'Sun flower', 'Coffee', 'Tobbaco', 'Casava', 'Ground nuts', 'S. Potatoes', 'Pawpaw', 'Jackfruit'], ['Avocado', 'Sun flower', 'Coffee all', 'Tobacco', 'Cassava', 'Groundnuts', 'Sweet potatoes', 'Paw paw', 'Jackfruit'], inplace=True)
ag5a.replace(['Bananaf','Bananan F','Bananas'], 'Banana Food', inplace=True)
ag5a.replace(['Ccffee', 'Coffee all'],'Coffee All',inplace=True)
ag5a.replace(['cowpeas'], 'Cow Peas', inplace=True)
ag5a.replace(['Eggplant'], 'Eggplants', inplace=True)
ag5a.replace(['Fieldpeas'], 'Field Peas', inplace=True)
ag5a.replace(['G.Nuts', 'Ground Nuts'], 'Groundnuts', inplace=True)
ag5a.replace(['Pigeon Pea', 'Peas'], 'Pigeon Peas', inplace=True)
ag5a.replace(['Soghurm'], 'Sorghum', inplace=True)
ag5a.replace(['Soya', 'Soyabean', 'Soyabeans'], 'Soya Beans', inplace=True)
ag5a.replace(['S.Potatoes', 'Sweet Potatoees', 'Sweet Potatoes620', 'Sweetpotatoes'], 'Sweet Potatoes', inplace=True)

lele = ag5a['cropID'].value_counts()

ag5a.loc[ag5a.unit==1, "tokg"] = 1

# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5a.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv(folder+'kg_conversion_09a.csv', index=False)
'''
### USE MEDIAN CONVERSION RATES ACROSS WAVES AND SEASONS
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_09']]
ag5a.replace(99999,np.nan, inplace=True)

ag5a.reset_index(inplace=True)
ag5a.rename(columns={'index':'i'}, inplace=True)
ag5a = ag5a.merge(kg_units, on=['unit','cropID'], how='left')
ag5a.drop_duplicates(subset=['i'], inplace=True) 


ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'kgconverter_09'] = ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'tokg']


ag5a['kgconverter_09'] = ag5a['kgconverter_09'].fillna(ag5a['tokg'])

# Convert to kg
ag5a[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5a["kgconverter_09"], axis="index")

for var in ['sell_kg', 'cons_kg', 'stored_kg']:
    ag5a.loc[ag5a[var]>ag5a['total_kg'], var] = ag5a.loc[ag5a[var]>ag5a['total_kg'], 'total_kg']

for var in ['gift_kg', 'food_prod_kg', 'animal_kg', 'seeds_kg']:
    ag5a.loc[ag5a[var]>ag5a['total_kg'], var] = ag5a.loc[ag5a[var]>ag5a['total_kg'], 'total_kg']


ag5a["total2_kg"] =  ag5a.loc[:,["sell_kg","gift_kg","cons_kg","food_prod_kg","animal_kg", "seeds_kg", "stored_kg"]].sum(axis=1)
ag5a["total2_kg"].replace(0,np.nan, inplace=True)


ag5a['total2_kg'] = ag5a['total2_kg'].fillna(ag5a['total_kg'])  #still many missing values 5259
#ag5a['total_kg'] = ag5a['total_kg'].fillna(ag5a['total2_kg'])

check = ag5a[['total_kg', 'total2_kg']].corr()
#  0.99 correlation and almost same distribution. Good!!

check_ag5a_kg = (ag5a[['HHID','total_kg','total2_kg']].groupby(by='HHID').sum()).describe(percentiles)


# yet let's replace the few ones that reported quantities without specifying unit. Assume they reported on kg.
for var in ["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]:
    ag5a[[var+'_kg']] = ag5a[[var+'_kg']].fillna(ag5a[[var]])


#### Prices
ag5a = ag5a.merge(basic, on='HHID', how='left')
ag5a["hh_price"] = ag5a.value_sells.div(ag5a.sell_kg, axis=0) 
# Set ipython's max row display

ag5a[['hh_price']] = ag5a[['cropID','hh_price']].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.975))


prices_nat = ag5a.groupby(by=["cropID"])[["hh_price"]].median()
prices_nat.columns=["p_nat"]
prices_nat[prices_nat>10000] = np.nan
prices_nat[prices_nat<50] = np.nan
prices_reg = ag5a.groupby(by=["cropID",'region'])[["hh_price"]].median()
prices_reg.columns=["p_reg"]
prices_reg[prices_reg>10000] = np.nan
prices_reg[prices_reg<50] = np.nan
prices_reg.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_county = ag5a.groupby(by=["cropID",'county'])[["hh_price"]].median()
prices_county.columns=["p_county"]
prices_county.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_county[prices_county>10000] = np.nan
prices_county[prices_county<50] = np.nan

prices_district = ag5a.groupby(by=["cropID",'district'])[["hh_price"]].median()
prices_district.columns=["p_district"]
prices_district.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_district[prices_district>10000] = np.nan
prices_district[prices_district<50] = np.nan


## Use consumption prices

## cons09.py needs to be run before

cprices_nat = pd.read_csv(folder+"pricesfood09.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood09.csv")
cprices_county = pd.read_csv(folder+"countypricesfood09.csv")

cprices_district = pd.read_csv(folder+"districtpricesfood09.csv")

for cprices in [cprices_nat, cprices_reg, cprices_county, cprices_district]:
    
    cprices["cropID"] = "nothing"
    cprices.loc[cprices["code"]==101,"cropID"] = "Banana Food"
    cprices.loc[cprices["code"]==105,"cropID"] = "Sweet Potatoes"
    cprices.loc[cprices["code"]==107,"cropID"] = "Cassava"
    cprices.loc[cprices["code"]==109,"cropID"] = 'Irish Potatoes'
    cprices.loc[cprices["code"]==110,"cropID"] = "Rice"
    cprices.loc[cprices["code"]==112,"cropID"] = "Maize"
    cprices.loc[cprices["code"]==115, "cropID"] = "Finger Millet"
    cprices.loc[cprices["code"]==116,"cropID"] = "Sorghum"
    cprices.loc[cprices["code"]==146,"cropID"] = "Simsim"
    cprices.loc[cprices["code"]==137,"cropID"] = "Cabbages"
    cprices.loc[cprices["code"]==138,"cropID"] = "Dodo"
    cprices.loc[cprices["code"]==136,"cropID"] = "Tomatoes"
    cprices.loc[cprices["code"]==135,"cropID"] = "Onions"
    cprices.loc[cprices["code"]==165,"cropID"] = "Pumpkins"
    cprices.loc[cprices["code"]==168,"cropID"] = "Eggplants"
    cprices.loc[cprices["code"]==170,"cropID"] = "Pineapples"
    cprices.loc[cprices["code"]==132,"cropID"] = 'Banana Sweet'
    cprices.loc[cprices["code"]==132,"cropID"] = "Mango"
    cprices.loc[cprices["code"]==130,"cropID"] = "Passion Fruit"
    cprices.loc[cprices["code"]==166,"cropID"] = "Avocado"
    cprices.loc[cprices["code"]==140,"cropID"] = "Beans"
    cprices.loc[cprices["code"]==144,"cropID"] = "Groundnuts"
    cprices.loc[cprices["code"]==145,"cropID"] = "Pigeon Peas"
    cprices.loc[cprices["code"]==162,"cropID"] = "Field Peas"
    cprices.loc[cprices["code"]==133,"cropID"] = "Oranges"
    cprices.loc[cprices["code"]== 171,"cropID"] = 'Paw paw' 

    
    cprices.drop(["code"], axis=1, inplace=True)


 # Set of prices
prices_nat = prices_nat.merge(cprices_nat, on="cropID", how="left")

#For the items that do not have consumption prices input the selling ones and viceversa.
prices_nat["m_p"] = prices_nat["m_p"].fillna(prices_nat["p_nat"])
prices_nat["p_nat"] = prices_nat["p_nat"].fillna(prices_nat["m_p"])
prices_nat.columns = ["cropID","p_sell_nat", "p_c_nat"]

## add reasonable prices for those missing
prices_nat.reset_index(inplace=True)
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_sell_nat"]].iloc[0,0]


prices_reg = prices_reg.merge(cprices_reg, on=["cropID", "region"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_reg["m_p"] = prices_reg["m_p"].fillna(prices_reg["p_reg"])
prices_reg["p_reg"] = prices_reg["p_reg"].fillna(prices_reg["m_p"])
prices_reg.columns = ["cropID",'region',"p_sell_reg", "p_c_reg"]

prices_county = prices_county.merge(cprices_county, on=["cropID", "county"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_county["m_p"] = prices_county["m_p"].fillna(prices_county["p_county"])
prices_county["p_county"] = prices_county["p_county"].fillna(prices_county["m_p"])
prices_county.columns = ["cropID",'county',"p_sell_county", "p_c_county"]

prices_district = prices_district.merge(cprices_district, on=["cropID", "district"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_district["m_p"] = prices_district["m_p"].fillna(prices_district["p_district"])
prices_district["p_district"] = prices_district["p_district"].fillna(prices_district["m_p"])
prices_district.columns = ["cropID",'district',"p_sell_district", "p_c_district"]

ag5a = ag5a.merge(prices_nat, on="cropID", how="left")
ag5a= ag5a.merge(prices_reg, on=["cropID", 'region'], how="left")
ag5a = ag5a.merge(prices_county, on=["cropID", 'county'], how="left")
ag5a = ag5a.merge(prices_district, on=["cropID", 'district'], how="left")

for price in ['p_sell', 'p_c']:
    ag5a[price+'_reg'].fillna(ag5a[price+'_nat'], inplace=True)
    ag5a[price+'_district'].fillna(ag5a[price+'_reg'], inplace=True)
    ag5a[price+'_county'].fillna(ag5a[price+'_district'], inplace=True)


quant = ["total", "sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county", "p_sell_district", "p_c_district"] 
values_ag5a = ag5a[["HHID", 'pltid', 'cropID',"total_kg","total2_kg","sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg", "trans_cost",'value_sells']]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q+'_kg']*ag5a[p]
        
(values_ag5a['total_value_p_c_nat'].isna()).sum()

for p in ['p_sell', 'p_c']:
    for area in ['_nat', '_reg', '_county', '_district']:
        ## Using each household reported value of sells plus using p_area to value non-sold production
        values_ag5a["total2_value_"+p+area] =  values_ag5a.loc[:,['value_sells',"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5a["total2_value_"+p+area] = values_ag5a["total2_value_"+p+area].replace(0, np.nan)
        
        ## Using median selling prices to compute sold value plus using p_area to value non-sold production
        values_ag5a["total3_value_"+p+area] =  values_ag5a.loc[:,["sell_value_p_sell"+area,"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5a["total3_value_"+p+area] = values_ag5a["total3_value_"+p+area].replace(0, np.nan)
        
        
(values_ag5a[['total_value_p_c_reg','total2_value_p_c_reg','total3_value_p_c_reg']].isna()).sum()


check = (values_ag5a[['HHID','total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat','total_value_p_c_reg','total2_value_p_c_reg','total3_value_p_c_reg']].groupby(by='HHID').sum()).replace(0,np.nan)/dollars
sum_ag5avalues = check.describe()

ag5a = values_ag5a
ag5a.reset_index(inplace=True)
ag5a.rename(columns={'index':'j'}, inplace=True)

# Merge datasets -------------------------------------------
agrica = pd.merge(agrica, ag5a, on=["HHID",  'pltid','cropID'], how='right')
#agrica.drop_duplicates(subset=['j'],inplace=True)
agrica.set_index(['HHID','pltid','cropID'], inplace=True)
agrica = agrica.reset_index()

#agrica[['org_fert', 'chem_fert', 'pesticides','seed_cost']] = remove_outliers(agrica[['org_fert', 'chem_fert', 'pesticides','seed_cost']],hq=0.95)

#for p in priceslist:
 #   agrica[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]] = remove_outliers(agrica[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]],lq=lq, hq=0.995)

count_crops = pd.value_counts(agrica['cropID'])



agrica['A'] = (agrica['area_planted']).replace(0,np.nan)
agrica['y'] = (agrica['total2_value_p_c_district']).replace(0,np.nan)
agrica['season'] = 1

agrica['y'].describe()
agrica_hh = (agrica.groupby('HHID').sum()).replace(0,np.nan)
sum_agrica = agrica_hh[['y']].describe(percentiles=percentiles)/dollars
sum_agrica_land = agrica_hh[['A']].describe(percentiles=percentiles)


#%% AGRICULTURAL SEASON 2:

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3b = pd.read_stata(my_dirct+'agsec3b.dta')
ag3b = ag3b[["HHID", 'a3bq3', 'a3bq5', "a3bq7", "a3bq8",  "a3bq16", "a3bq18", "a3bq19", 'a3bq30',  'a3bq28b', 'a3bq31', 'a3bq38', 'a3bq39', 'a3bq42a', 'a3bq42b','a3bq42c', 'a3bq43']]
ag3b['hhlabor'] = ag3b["a3bq39"].fillna(0) 
ag3b['hired_labor'] = ag3b["a3bq42a"].fillna(0)+ag3b["a3bq42b"].fillna(0)+ag3b["a3bq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3b['p_orgfert'] = np.nanmedian( ag3b['a3bq8'].div(ag3b['a3bq7'], axis=0) )
ag3b['p_chemfert'] = np.nanmedian( ag3b['a3bq19'].div(ag3b['a3bq18'], axis=0) )
ag3b['p_pest'] = np.nanmedian( ag3b['a3bq31'].div(ag3b['a3bq30'], axis=0) )  ## missing quantity pesticies purchased in the data (but in the questionnaire is there)

ag3b['org_fert'] = ag3b['p_orgfert']*ag3b['a3bq5']
### important changes when included non-bought 
ag3b['org_fert'].describe()
ag3b['a3bq8'].describe()

## with chemfert might not be important: indeed same number of obs.
ag3b['chem_fert'] = ag3b['p_chemfert']*ag3b['a3bq16']
ag3b['chem_fert'].describe()
ag3b['a3bq18'].describe()

##pesticides: small changes
ag3b['pesticides'] = ag3b['p_pest']*ag3b['a3bq28b']
ag3b['pesticides'].describe()
ag3b['a3bq31'].describe()


ag3b = ag3b[["HHID", 'a3bq3', "org_fert", "chem_fert", 'pesticides','hhlabor', 'hired_labor',"a3bq43"]]
ag3b.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']

# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4b = pd.read_stata(my_dirct+'agsec4b.dta')


ag4b = ag4b[["HHID", 'a4bq2','a4bq6' , 'a4bq8', 'a4bq9',  "a4bq11"]]
ag4b.columns = ["HHID", 'pltid', 'crop_code', 'total_area', 'weightcrop',  'seed_cost']
ag4b['weightcrop'].replace(np.nan,100, inplace=True)



### I don't have cropID.. get it from season A
crop_codes = ag4a[['crop_code','cropID']].drop_duplicates()
ag4b = pd.merge(ag4b, crop_codes, on=['crop_code'], how='left')
ag4b.drop_duplicates(subset=['HHID','pltid','total_area','crop_code'], inplace=True)
ag4b = ag4b.dropna(subset=['total_area'])
b = pd.value_counts(ag4b['crop_code'])
a = pd.value_counts(ag4a['crop_code'])


### already ask total area and proportion per crop
ag4b['area_planted'] = ag4b['total_area'].multiply(ag4b['weightcrop']/100, axis=0)

ag4b.reset_index(inplace=True)
ag4b.rename(columns={'index':'i'}, inplace=True)

agricb = pd.merge(ag3b, ag4b, on=['HHID','pltid'], how='right')
agricb.drop_duplicates(subset='i', inplace=True)


### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agricb['weightcrop']/100, axis=0)

#COST


# =============================================================================
# Output
# =============================================================================
ag5b = pd.read_stata(my_dirct+'agsec5b.dta')
ag5b = ag5b[["HHID",'a5bq3',"a5bq4","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq22"]]
ag5b.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]

ag5b['cropID'] = ag5b['cropID'].str.title()
ag5b.replace(['Avacoda', 'Sun flower', 'Coffee', 'Tobbaco', 'CASAVA', 'Casava', 'Ground nuts', 'S. Potatoes', 'Paw paw', 'Pawpaws', 'Jackfruit'], ['Avocado', 'Sunflower', 'Coffee all', 'Tobacco', 'Cassava','Cassava', 'Groundnuts', 'Sweet potatoes', 'Paw Paw', 'Paw Paw', 'Jack Fruit'], inplace=True)
ag5b.replace(['Bananaf','Bananan F','Bananas', 'Banana'], 'Banana Food', inplace=True)
ag5b.replace(['Ccffee', 'Coffee all'],'Coffee All',inplace=True)
ag5b.replace(['cowpeas', 'Cowpeas'], 'Cow Peas', inplace=True)
ag5b.replace(['Eggplant'], 'Eggplants', inplace=True)
ag5b.replace(['Fieldpeas'], 'Field Peas', inplace=True)
ag5b.replace(['G.Nuts', 'Ground Nuts'], 'Groundnuts', inplace=True)
ag5b.replace(['Pigeon Pea', 'Peas'], 'Pigeon Peas', inplace=True)
ag5b.replace(['Soghurm'], 'Sorghum', inplace=True)
ag5b.replace(['Soya', 'Soyabean', 'Soyabeans'], 'Soya Beans', inplace=True)
ag5b.replace(['S.Potatoes', 'Sweet Potatoees', 'Sweet Potatoes620', 'Sweetpotatoes', 'Lumonde'], 'Sweet Potatoes', inplace=True)
ag5b.replace(['Matooke'], 'Banana Food', inplace=True)
ag5b.replace(['Irishpotatoes', 'Potatoes'], 'Irish Potatoes', inplace=True)
ag5b.replace(['S.Cane','Surgarcane'], 'Sugarcane', inplace=True)
ag5b.replace(['Mangoes','Yams','Tomato', 'Pineapple'], ['Mango','Yam','Tomatoes', 'Pineapples'], inplace=True)
ag5b.replace(['Vanila'], 'Vanilla', inplace=True)
ag5b.replace(['Sweet Bananas'], 'Banana Sweet', inplace=True)
#ag5b.replace(['Millet'], 'Finger Millet', inplace=True)

ag5b.loc[ag5b.unit==1, "tokg"] = 1
# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5b.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv(folder+'kg_conversion_09b.csv', index=False)
'''

### USE MEDIAN CONVERSION RATES ACROSS WAVES AND SEASONS
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_09']]
ag5b.replace(99999,np.nan, inplace=True)
ag5b.reset_index(inplace=True)
ag5b.rename(columns={'index':'i'}, inplace=True)

ag5b = ag5b.merge(kg_units, on=['unit','cropID'], how='left')
ag5b.drop_duplicates(subset=['i'], inplace=True) 

ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'kgconverter_09'] = ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'tokg']


ag5b['kgconverter_09'] = ag5b['kgconverter_09'].fillna(ag5b['tokg'])

# Convert to kg
ag5b[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5b["kgconverter_09"], axis="index")

for var in ['sell_kg', 'cons_kg', 'stored_kg']:
    ag5b.loc[ag5b[var]>ag5b['total_kg'], var] = ag5b.loc[ag5b[var]>ag5b['total_kg'], 'total_kg']

for var in ['gift_kg', 'food_prod_kg', 'animal_kg', 'seeds_kg']:
    ag5b.loc[ag5b[var]>ag5b['total_kg'], var] = ag5b.loc[ag5b[var]>ag5b['total_kg'], 'total_kg']


ag5b["total2_kg"] =  ag5b.loc[:,["sell_kg","gift_kg","cons_kg","food_prod_kg","animal_kg", "seeds_kg", "stored_kg"]].sum(axis=1)
ag5b["total2_kg"].replace(0,np.nan, inplace=True)


ag5b['total2_kg'] = ag5b['total2_kg'].fillna(ag5b['total_kg'])  #still many missing values 5259
#ag5b['total_kg'] = ag5b['total_kg'].fillna(ag5b['total2_kg'])

check = ag5b[['total_kg', 'total2_kg']].corr()
check_ag5b_kg = (ag5b[['HHID','total_kg','total2_kg']].groupby(by='HHID').sum()).describe(percentiles)
#  0.99 correlation and almost same distribution. Good!!


# yet let's replace the few ones that reported quantities without specifying unit. Assume they reported on kg.
for var in ["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]:
    ag5b[[var+'_kg']] = ag5b[[var+'_kg']].fillna(ag5b[[var]])


#### Prices
ag5b = ag5b.merge(basic, on='HHID', how='left')
ag5b["hh_price"] = ag5b.value_sells.div(ag5b.sell, axis=0) 

#ag5b[['hh_price']] = ag5b[['cropID','hh_price']].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.975))


prices_nat = ag5b.groupby(by=["cropID"])[["hh_price"]].median()
prices_nat.columns=["p_nat"]
prices_nat[prices_nat>10000] = np.nan
prices_nat[prices_nat<50] = np.nan
prices_reg = ag5b.groupby(by=["cropID",'region'])[["hh_price"]].median()
prices_reg.columns=["p_reg"]
prices_reg[prices_reg>10000] = np.nan
prices_reg[prices_reg<50] = np.nan
prices_reg.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_county = ag5b.groupby(by=["cropID",'county'])[["hh_price"]].median()
prices_county.columns=["p_county"]
prices_county.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_county[prices_county>10000] = np.nan
prices_county[prices_county<50] = np.nan

prices_district = ag5b.groupby(by=["cropID",'district'])[["hh_price"]].median()
prices_district.columns=["p_district"]
prices_district.replace([-np.inf, +np.inf],np.nan, inplace=True)
prices_district[prices_district>10000] = np.nan
prices_district[prices_district<50] = np.nan


## Use consumption prices
cprices_nat = pd.read_csv(folder+"pricesfood09.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood09.csv")
cprices_county = pd.read_csv(folder+"countypricesfood09.csv")
cprices_district = pd.read_csv(folder+"districtpricesfood09.csv")

for cprices in [cprices_nat, cprices_reg, cprices_county, cprices_district]:
    
    cprices["cropID"] = "nothing"
    cprices.loc[cprices["code"]==101,"cropID"] = "Banana Food"
    cprices.loc[cprices["code"]==105,"cropID"] = "Sweet Potatoes"
    cprices.loc[cprices["code"]==107,"cropID"] = "Cassava"
    cprices.loc[cprices["code"]==109,"cropID"] = 'Irish Potatoes'
    cprices.loc[cprices["code"]==110,"cropID"] = "Rice"
    cprices.loc[cprices["code"]==112,"cropID"] = "Maize"
    cprices.loc[cprices["code"]==115, "cropID"] = "Finger Millet"
    cprices.loc[cprices["code"]==116,"cropID"] = "Sorghum"
    cprices.loc[cprices["code"]==146,"cropID"] = "Simsim"
    cprices.loc[cprices["code"]==137,"cropID"] = "Cabbages"
    cprices.loc[cprices["code"]==138,"cropID"] = "Dodo"
    cprices.loc[cprices["code"]==136,"cropID"] = "Tomatoes"
    cprices.loc[cprices["code"]==135,"cropID"] = "Onions"
    cprices.loc[cprices["code"]==165,"cropID"] = "Pumpkins"
    cprices.loc[cprices["code"]==168,"cropID"] = "Eggplants"
    cprices.loc[cprices["code"]==170,"cropID"] = "Pineapples"
    cprices.loc[cprices["code"]==132,"cropID"] = 'Banana Sweet'
    cprices.loc[cprices["code"]==132,"cropID"] = "Mango"
    cprices.loc[cprices["code"]==130,"cropID"] = "Passion Fruit"
    cprices.loc[cprices["code"]==166,"cropID"] = "Avocado"
    cprices.loc[cprices["code"]==140,"cropID"] = "Beans"
    cprices.loc[cprices["code"]==144,"cropID"] = "Groundnuts"
    cprices.loc[cprices["code"]==145,"cropID"] = "Pigeon Peas"
    cprices.loc[cprices["code"]==162,"cropID"] = "Field Peas"
    cprices.loc[cprices["code"]==133,"cropID"] = "Oranges"
    cprices.loc[cprices["code"]== 171,"cropID"] = 'Paw paw' 

    
    cprices.drop(["code"], axis=1, inplace=True)


# Set of prices
prices_nat = prices_nat.merge(cprices_nat, on="cropID", how="left")

#For the items that do not have consumption prices input the selling ones and viceversa.
prices_nat["m_p"] = prices_nat["m_p"].fillna(prices_nat["p_nat"])
prices_nat["p_nat"] = prices_nat["p_nat"].fillna(prices_nat["m_p"])
prices_nat.columns = ["cropID","p_sell_nat", "p_c_nat"]

## add reasonable prices for those missing

prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Millet', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_sell_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Millet', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_c_nat"]].iloc[0,0]


prices_nat.loc[prices_nat['cropID']=='Coffee All', ["p_c_nat"]] = 500  #previous season price
prices_nat.loc[prices_nat['cropID']=='Coffee All', ["p_sell_nat"]] = 500

prices_nat.loc[prices_nat['cropID']=='Soya Beans', ["p_c_nat"]] = 650  #previous season price
prices_nat.loc[prices_nat['cropID']=='Soya Beans', ["p_sell_nat"]] = 650


prices_nat.drop(54, inplace=True)

prices_reg = prices_reg.merge(cprices_reg, on=["cropID", "region"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_reg["m_p"] = prices_reg["m_p"].fillna(prices_reg["p_reg"])
prices_reg["p_reg"] = prices_reg["p_reg"].fillna(prices_reg["m_p"])
prices_reg.columns = ["cropID",'region',"p_sell_reg", "p_c_reg"]

prices_county = prices_county.merge(cprices_county, on=["cropID", "county"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_county["m_p"] = prices_county["m_p"].fillna(prices_county["p_county"])
prices_county["p_county"] = prices_county["p_county"].fillna(prices_county["m_p"])
prices_county.columns = ["cropID",'county',"p_sell_county", "p_c_county"]

prices_district = prices_district.merge(cprices_district, on=["cropID", "district"], how="left")
#For the items that do not have consumption prices input the selling ones and viceversa.
prices_district["m_p"] = prices_district["m_p"].fillna(prices_district["p_district"])
prices_district["p_district"] = prices_district["p_district"].fillna(prices_district["m_p"])
prices_district.columns = ["cropID",'district',"p_sell_district", "p_c_district"]

ag5b = ag5b.merge(prices_nat, on="cropID", how="left")
ag5b= ag5b.merge(prices_reg, on=["cropID", 'region'], how="left")
ag5b = ag5b.merge(prices_county, on=["cropID", 'county'], how="left")
ag5b = ag5b.merge(prices_district, on=["cropID", 'district'], how="left")


for price in ['p_sell', 'p_c']:
    ag5b[price+'_reg'].fillna(ag5b[price+'_nat'], inplace=True)
    ag5b[price+'_district'].fillna(ag5b[price+'_reg'], inplace=True)
    ag5b[price+'_county'].fillna(ag5b[price+'_district'], inplace=True)


quant = ["total", "sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 
values_ag5b = ag5b[["HHID", 'pltid', 'cropID',"total_kg","total2_kg","sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg", "trans_cost",'value_sells']]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5b[q+"_value_"+p] = ag5b[q+'_kg']*ag5b[p]
        
(values_ag5b['total_value_p_c_nat'].isna()).sum()

for p in ['p_sell', 'p_c']:
    for area in ['_nat', '_reg', '_county', '_district']:
        ## Using each household reported value of sells plus using p_area to value non-sold production
        values_ag5b["total2_value_"+p+area] =  values_ag5b.loc[:,['value_sells',"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5b["total2_value_"+p+area].replace(0, np.nan, inplace=True)
        values_ag5b["total2_value_"+p+area].fillna(values_ag5b["total_value_"+p+area], inplace=True)
        
        ## Using median selling prices to compute sold value plus using p_area to value non-sold production
        values_ag5b["total3_value_"+p+area] =  values_ag5b.loc[:,["sell_value_p_sell"+area,"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5b["total3_value_"+p+area].replace(0, np.nan, inplace=True)
        values_ag5b["total3_value_"+p+area].fillna(values_ag5b["total_value_"+p+area], inplace=True)
        
(values_ag5b[['total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat']].isna()).sum()


check = (values_ag5b[['HHID','total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat']].groupby(by='HHID').sum()).replace(0,np.nan)/dollars
sum_ag5bvalues = check.describe()

ag5b = values_ag5b
ag5b.reset_index(inplace=True)
ag5b.rename(columns={'index':'j'}, inplace=True)

# Merge datasets -------------------------------------------
agricb = pd.merge(agricb, ag5b, on=["HHID",  'pltid','cropID'], how='right')
agricb.drop_duplicates(subset=['j'],inplace=True)
agricb.set_index(['HHID','pltid','cropID'], inplace=True)
agricb = agricb.reset_index()


#agricb[['org_fert', 'chem_fert', 'pesticides','seed_cost']] = remove_outliers(agricb[['org_fert', 'chem_fert', 'pesticides','seed_cost']],hq=0.95)

#for p in priceslist:
 #   agricb[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]] = remove_outliers(agricb[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]],lq=lq, hq=0.998)


count_crops = pd.value_counts(agricb['cropID'])


agricb['A'] = (agricb['area_planted']).replace(0,np.nan)
agricb['y'] = (agricb['total2_value_p_c_district']).replace(0,np.nan)
agricb['season'] = 2


agricb_hh = (agricb.groupby('HHID').sum()).replace(0,np.nan)
sum_agricb = agricb_hh[['y']].describe(percentiles=percentiles)/dollars
sum_agricb_land = agricb_hh[['A']].describe(percentiles=percentiles)




#%% Livestock

#Big Animals------------------------------------------------------------
ag6a = pd.read_stata(my_dirct+'AGSEC6A.dta')
ag6a = ag6a[["HHID", "a6aq3", "a6aq5", "a6aq12", "a6aq13","a6aq14","a6aq15"]]
ag6a.columns = ['HHID',"lvstid","lvstk_big" ,"bought", "value_bought", "sold", "value_sold_big"]

ag6a['p_bought'] = ag6a['value_bought']/ag6a['bought']
ag6a['p_sold'] = ag6a['value_sold_big']/ag6a['sold']

#Obtain prices animals
prices = ag6a.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6a_2009.csv')
ag6a = pd.merge(ag6a, prices, on="lvstid", how='outer')

## Big livestock wealth--------------------------------------
ag6a['lvstk_big_value'] = ag6a.lvstk_big*ag6a.p_sold_y
ag6a = ag6a.groupby(by='HHID')[["value_sold_big", 'lvstk_big_value']].sum()



#Small animals=============================================================
ag6b = pd.read_stata(my_dirct+'AGSEC6B.dta', convert_categoricals=False)
ag6b = ag6b[["HHID", "a6bq3", "a6bq5", "a6bq12","a6bq13","a6bq14","a6bq15"]]
ag6b.columns = ["HHID", "lvstid", "lvstk_small", "bought", "value_bought", "sold", "value_sold_small"]

ag6b['p_bought'] = ag6b['value_bought']/ag6b['bought']
ag6b['p_sold'] = ag6b['value_sold_small']/ag6b['sold']


## Obtain prices --------------------------
prices = ag6b.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6b_2009.csv')
ag6b = pd.merge(ag6b, prices, on="lvstid", how='outer')

## Income: report sellings in last 6 months
ag6b['value_sold_small'] = 2*ag6b['value_sold_small'] 
## small livestock wealth--------------------------------------
ag6b['lvstk_small_value'] = ag6b.lvstk_small*ag6b.p_sold_y

ag6b = ag6b.groupby(by='HHID')[["value_sold_small", 'lvstk_small_value']].sum()



#Poultry animals==========================================
ag6c = pd.read_stata(my_dirct+'AGSEC6C.dta' )
ag6c = ag6c[["HHID", "a6cq3", "a6cq5","a6cq12","a6cq13","a6cq14", "a6cq15"]]
ag6c.columns = ['HHID',"lvstid", "lvstk_poultry", "bought", "value_bought", "sold", "value_sold_poultry"]

ag6c['p_bought'] = ag6c['value_bought']/ag6c['bought']
ag6c['p_sold'] = ag6c['value_sold_poultry']/ag6c['sold']

## Obtain prices --------------------------
prices = ag6c.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6c_2009.csv')
ag6c = pd.merge(ag6c, prices, on="lvstid", how='outer')


# Income selling poultry livestock-----------------------
#Ask last 3 months
ag6c["value_sold_poultry"] = 4*ag6c.value_sold_poultry

## poultry livestock wealth--------------------------------------
ag6c['lvstk_poultry_value'] = ag6c.lvstk_poultry*ag6c.p_sold_y
ag6c = ag6c.groupby(by='HHID')[["value_sold_poultry", 'lvstk_poultry_value']].sum()


# Livestock inputs----------------------
ag7 = pd.read_stata(my_dirct+'AGSEC7.dta')
ag7 = ag7[["HHID", "a7q4"]]
ag7 = ag7.groupby(by="HHID").sum()
ag7.reset_index(inplace=True)
ag7.columns = ['HHID',"animal_inp"]

#COST


# Livestock Outputs-------------------------------------------------------------
ag8 = pd.read_stata(my_dirct+'AGSEC8.dta')
ag8 = ag8[["HHID","a8q7", "a8q8"]]
ag8.columns = ["HHID", "month_sales","month_c"]
ag8["lvst_output"] = ag8[["month_sales"]]*12
ag8["animal_c"] = ag8[["month_c"]]*12
ag8 = ag8.groupby(by="HHID")[["lvst_output"]].sum()
ag8.reset_index(inplace=True)


#Extension service---------------------------------------------------
ag9 = pd.read_stata(my_dirct+'AGSEC9A.dta')
ag9 = ag9[["HHID", "a9q2","a9q9"]]
ag9.columns = ["HHID", "a9q2", "consulting_cost"]
values = [ "consulting_cost"]
index = ['HHID', "a9q2"]
panel = ag9.pivot_table(values=values, index=index)
ag9 = panel.sum(axis=0, level="HHID")


#Machinery-----------------------------------------------------------
## NO MACHINERY. FROM THE DICTIONARY IN THE WORLDBANK WEBSITE THERE SHOULD BE BUT THE DATASET AG10 DOES NOT CONTAIN IT (IT IS A DIFFERENT DATASET)


#Merge datasets------------------------------------------------------
livestock = pd.merge(ag6a, ag6b, on='HHID', how='outer')
livestock = pd.merge(livestock, ag6c, on='HHID', how='outer')
livestock = pd.merge(livestock, ag7, on='HHID', how='outer')
livestock = pd.merge(livestock, ag8, on='HHID', how='outer')
livestock = pd.merge(livestock, ag9, on='HHID', how='outer')

livestock["revenue_lvstk"] =livestock.loc[:,["value_sold_big","value_sold_small","value_sold_poultry","lvst_output"]].sum(axis=1) 
livestock["revenue_lvstk"] = livestock["revenue_lvstk"].replace(0,np.nan)
livestock["cost_lvstk"] = livestock.loc[:,["animal_inp"]].sum(axis=1) 
livestock["cost_lvstk"] = livestock["cost_lvstk"].replace(0,np.nan)
livestock["wealth_lvstk"] = livestock.loc[:,["lvstk_big_value","lvstk_small_value","lvstk_poultry_value"]].sum(axis=1) 
livestock["wealth_lvstk"] = livestock["wealth_lvstk"].replace(0,np.nan)

ls = livestock[["HHID","revenue_lvstk", "cost_lvstk", "wealth_lvstk"]]
ls = ls.dropna()

ls["HHID"] = pd.to_numeric(ls['HHID'])



ls["profit_lvstk"] = ls["revenue_lvstk"].fillna(0) - ls["cost_lvstk"].fillna(0)

sumls = ls.describe()/dollars


#%% Merge Seasons

data = agrica.append(agricb)
data.reset_index(inplace=True)
count_crops = pd.value_counts(data['cropID'])
data['HHID'] = pd.to_numeric(data["HHID"])


#### DATA AT HOUSEHOLD LEVEL TO COMPUTE HOUSEHOLD INCOME
data_hh = data.groupby(by="HHID").sum()  # we careful some variables doesnt make sense to sum (like land)
data_hh = data_hh.merge(ag2a, on='HHID', how='left')
data_hh = data_hh.merge(ag2b, on='HHID', how='left')
data_hh = pd.merge(data_hh, ls, on="HHID", how="outer")

data_hh.rename(columns={"HHID": "hh"}, inplace=True)

data_hh['A'] = data_hh['A']/2
data_hh['m'] = data_hh[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)


data_hh['cost_agr'] = -data_hh[['m']]
for p in priceslist:
    data_hh['profit_agr_'+p] = data_hh.loc[:,['total2_value_'+p,"cost_agr"]].sum(axis=1)    
    data_hh['revenue_agr_'+p] = data_hh["total2_value_"+p]
   
sumdata_hh = data_hh[['total2_value_p_c_nat','total2_value_p_c_reg', 'total2_value_p_c_county', 'total2_value_p_c_district',
                       'total2_value_p_c_nat', 'total3_value_p_c_reg', 'total3_value_p_c_county', 'total3_value_p_c_district',
                       'total2_value_p_sell_nat','total2_value_p_sell_reg', 'total2_value_p_sell_county',  'total2_value_p_sell_district', 
                       'total2_value_p_sell_nat', 'total3_value_p_sell_reg', 'total3_value_p_sell_county', 'total3_value_p_sell_district']].describe()/dollars

#### agrls Wealth:
data_hh["wealth_agrls"] = data_hh.loc[:,["wealth_lvstk"]].sum(axis=1)
data_hh["wealth_agrls"] = data_hh["wealth_agrls"].replace(0,np.nan)

wealth = data_hh[["hh","wealth_agrls","wealth_lvstk"]]


### Agricultural wealth and income to compute total household W and I
if save==True:
    wealth.to_csv(folder+"wealth_agrls09.csv", index=False)
del data_hh['wealth_agrls'], data_hh['wealth_lvstk']
if save==True:
    data_hh.to_csv(folder+"inc_agsec09.csv", index=False)



data['m'] = data[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)
data['l'] = data[['hhlabor','hired_labor']].sum(axis=1)


## DATA AT HHID-PLOT-CROP-SEASON LEVEL TO CROP PRODUCTIVITY STUDY:
#### Remove outliers at crop-season level-------------------------
#data[['y','m']] = data[['cropID','y','m']].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.995))
#data[['A']] = data[['A','cropID']].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.995))


#### For data at plot-crop level, contro for inflation ------------------
basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","region","urban"]] 
basic['HHID'] = pd.to_numeric(basic['HHID'])
data = pd.merge(data, basic, on='HHID', how='left')

data['inflation_avg'] =  0.709890102

#Divide by inflation all monetary variables: .div(data.inflation, axis=0)
data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']] = data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']].div(data.inflation_avg, axis=0)/(dollars)

priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 
for p in priceslist:
    data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "gift_value_"+p]] = data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p,"sell_value_"+p, "cons_value_"+p, "stored_value_"+p,"gift_value_"+p]].div(data.inflation_avg, axis=0)/(dollars)

data_check = (data.groupby(by="HHID").sum()).replace(0,np.nan)
data_check.reset_index(inplace=True)
data_check['A'] = data_check['A']/2

sumdatahh = data_check[['y','m','A','l','total_value_p_c_reg','total2_value_p_c_reg', 'total2_value_p_c_county',  'total2_value_p_c_district', 
                        'total3_value_p_c_reg', 'total3_value_p_c_county',  'total3_value_p_c_district' ]].describe(percentiles=percentiles)

# the maximums of the variables seem okay. no need trimming.

outliers = data_check.loc[(data_check['y']>np.nanpercentile(data_check['y'], 99)) | (data_check['A']>np.nanpercentile(data_check['A'],100)),'HHID']

data = data[~data['HHID'].isin(outliers)]

data_check = data.groupby(by="HHID").sum()
data_check.reset_index(inplace=True)
data_check['A'] = data_check['A']/2

sumdatahh2 = data_check[['y','m','A','l','total_value_p_c_reg','total2_value_p_c_reg', 'total2_value_p_c_county',  'total2_value_p_c_district', 
                        'total3_value_p_c_reg', 'total3_value_p_c_county',  'total3_value_p_c_district' ]].describe(percentiles=percentiles)

sum_m = data_check[['m', 'chem_fert','org_fert','pesticides', 'seed_cost','trans_cost','labor_payment']].describe(percentiles=percentiles)
### Data at the hh-plot-crop level to run analysis across crops
if save==True:
    data.to_csv(folder+'agric_data09.csv', index=False)
print('data 2009 agric saved')