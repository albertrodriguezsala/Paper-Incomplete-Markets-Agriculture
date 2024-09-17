# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:42:05 2018

@author: rodri
"""

# =============================================================================
# #### Agricultural data Uganda 2010-11
# =============================================================================


'''
DESCRIPTION:
    Uses the data from the 2010-2011 Integrated Survey in Agriculture from the UNPS (ISA-LSMS) to obtain:
        For the 2 main harvest of the year of the survey. (fall 2010 and spring 2011/)
            -agricultural inputs variables at plot level and household level.
            -agricultural ouptut variables at plot level, household level, and crop level.   
        -household land, household farming assets (capital) household level
        - Livestock revenues (including non-sold consumption), costs, and stock (wealth) household level
    output: inc_agsec10.csv (agric income and costs at hh level) wealth_agrls10.csv (agric wealth: livestock+farming capital), agric_data10.csv (outputs and inputs at crop-plot-hh level)
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

os.chdir(dirct)

my_dirct = str(dirct)+'/data/raw data/2010/'  
folder =  str(dirct)+'/data/data10/'
folder2 = str(dirct)+'/data/auxiliary data/'

pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89

hq=0.99
lq=0.00

hq_2 = 0.95


percentiles = [ 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )

basic = basic[["HHID","region","urban", 'h1aq1', "h1aq2b", "h1aq3b", "h1aq4b","year", "month"]] 
basic.columns = ["HHID","region","urban", 'district' , "county", 'subcounty', 'parish', "year", "month"]
basic['district'] = basic['district'].str.upper()
blu = basic['district'].value_counts()
bla = basic['county'].unique()

#%% AGRICULTURAL SEASON 1:


#rent obtained------------------------------
ag2a = pd.read_stata(my_dirct+'AGSEC2A.dta')
ag2a = ag2a[["HHID", "a2aq14"]]
ag2a = ag2a.groupby(by="HHID")[["a2aq14"]].sum()
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
ag3a = ag3a[["HHID",'prcid','pltid', 'a3aq5','a3aq7', "a3aq8", 'a3aq16',"a3aq18","a3aq19", 'a3aq28b', 'a3aq30','a3aq31', 'a3aq38', 'a3aq39', 'a3aq42a', 'a3aq42b','a3aq42c', 'a3aq43']]
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

ag3a = ag3a[["HHID",'prcid', 'pltid', "org_fert", "chem_fert", "pesticides",'hhlabor', 'hired_labor', "a3aq43"]]
ag3a.columns = ["HHID",'prcid', 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']



# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
crop_codes = pd.read_csv(folder2+'crop_codes.csv')
crop_codes.columns = ['itemID','cropID']
ag4a = pd.read_stata(my_dirct+'agsec4a.dta')
ag4a = ag4a[["HHID",'prcid', 'pltid','cropID' , 'a4aq8', 'a4aq9',  "a4aq11"]]
ag4a.columns = ["HHID",'prcid', 'pltid','itemID' , 'total_area', 'weightcrop', 'seed_cost']
ag4a = pd.merge(ag4a, crop_codes, on='itemID', how='left' )
ag4a['weightcrop'].replace(np.nan,100, inplace=True)

### already ask total area and proportion per crop
ag4a['area_planted'] = ag4a['total_area'].multiply(ag4a['weightcrop']/100, axis=0)

ag4a.reset_index(inplace=True)
ag4a.rename(columns={'index':'i'}, inplace=True)
agrica = pd.merge(ag3a, ag4a, on=['HHID','prcid','pltid'], how='right')
agrica.drop_duplicates(subset=['i'], inplace=True) 


### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agrica['weightcrop']/100, axis=0)


# =============================================================================
# Output
# =============================================================================

ag5a = pd.read_stata(my_dirct+'agsec5a.dta', convert_categoricals=False)
ag5a = ag5a[["HHID",'prcid','pltid',"cropID","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq22"]]
ag5a.columns = ["HHID",'prcid','pltid', "itemID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]
ag5a = pd.merge(ag5a, crop_codes, on='itemID', how='left' )

ag5a.loc[ag5a.unit==1, "tokg"] = 1
crops_unique = ag5a['cropID'].unique()

# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5a.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv('C:/Users/rodri/Dropbox/JMP/data/data10/kg_conversion_10a.csv',index=False)
'''

### USE MEDIAN CONVERSION RATES ACROSS WAVES AND SEASONS
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_10']]
kg_units.columns = ['unit','cropID','kgconverter_10']

ag5a.replace(99999,np.nan, inplace=True)

ag5a.reset_index(inplace=True)
ag5a.rename(columns={'index':'i'}, inplace=True)
ag5a = ag5a.merge(kg_units, on=['unit','cropID'], how='left')
ag5a.drop_duplicates(subset=['i'], inplace=True) 

ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'kgconverter_10'] = ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'tokg']

ag5a['kgconverter_10'] = ag5a['kgconverter_10'].fillna(ag5a['tokg'])

# Convert to kg
ag5a[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5a["kgconverter_10"], axis="index")

for var in ['sell_kg', 'cons_kg', 'stored_kg']:
    ag5a.loc[ag5a[var]>ag5a['total_kg'], var] = ag5a.loc[ag5a[var]>ag5a['total_kg'], 'total_kg']

for var in ['gift_kg', 'food_prod_kg', 'animal_kg', 'seeds_kg']:
    ag5a.loc[ag5a[var]>ag5a['total_kg'], var] = ag5a.loc[ag5a[var]>ag5a['total_kg'], 'total_kg']

ag5a["total2_kg"] =  ag5a.loc[:,["sell_kg","gift_kg","cons_kg","food_prod_kg","animal_kg", "seeds_kg", "stored_kg"]].sum(axis=1)
ag5a["total2_kg"].replace(0,np.nan, inplace=True)

ag5a['total2_kg'] = ag5a['total2_kg'].fillna(ag5a['total_kg'])  
ag5a['total_kg'] = ag5a['total_kg'].fillna(ag5a['total2_kg'])

check = ag5a[['total_kg', 'total2_kg']].corr()

check_ag5a_kg = (ag5a[['HHID','total_kg','total2_kg']].groupby(by='HHID').sum()).describe(percentiles)
#  0.82 correlation and almost same distribution. Good!!


# yet let's replace the few ones that reported quantities without specifying unit. Assume they reported on kg.
for var in ["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]:
    ag5a[[var+'_kg']] = ag5a[[var+'_kg']].fillna(ag5a[[var]])


#### Prices
ag5a = ag5a.merge(basic, on='HHID', how='left')
ag5a["hh_price"] = ag5a.value_sells.div(ag5a.sell_kg, axis=0) 
# Set ipython's max row display

#ag5a[['hh_price']] = ag5a[['cropID','hh_price']].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.975))


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
cprices_nat = pd.read_csv(folder+"pricesfood10.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood10.csv")
cprices_county = pd.read_csv(folder+"countypricesfood10.csv")
cprices_district = pd.read_csv(folder+"districtpricesfood10.csv")

for cprices in [cprices_nat, cprices_reg, cprices_county, cprices_district]:
    
    cprices["cropID"] = "nothing"
    cprices.loc[cprices["code"]==101,"cropID"] = "Banana Food"
    cprices.loc[cprices["code"]==105,"cropID"] = "Sweet Potatoes"
    cprices.loc[cprices["code"]==108,"cropID"] = "Cassava"
    cprices.loc[cprices["code"]==109,"cropID"] = 'Irish Potatoes'
    cprices.loc[cprices["code"]==110,"cropID"] = "Rice"
    cprices.loc[cprices["code"]==112,"cropID"] = "Maize"
    cprices.loc[cprices["code"]==115, "cropID"] = "Finger Millet"
    cprices.loc[cprices["code"]==116,"cropID"] = "Sorghum"
    cprices.loc[cprices["code"]==146,"cropID"] = "SimSim"
    cprices.loc[cprices["code"]==137,"cropID"] = "Cabbage"
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
    cprices.loc[cprices["code"]== 171,"cropID"] = 'Paw Paw'
    cprices.drop(["code"], axis=1, inplace=True)

 # Set of prices
prices_nat = prices_nat.merge(cprices_nat, on="cropID", how="left")

#For the items that do not have consumption prices input the selling ones and viceversa.
prices_nat["m_p"] = prices_nat["m_p"].fillna(prices_nat["p_nat"])
prices_nat["p_nat"] = prices_nat["p_nat"].fillna(prices_nat["m_p"])
prices_nat.columns = ["cropID","p_sell_nat", "p_c_nat"]

## add reasonable prices for those missing
prices_nat.loc[prices_nat['cropID']=='Wheat', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Wheat', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_sell_nat"]].iloc[0,0]


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
priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 
values_ag5a = ag5a[["HHID",'prcid', 'pltid', 'cropID',"total_kg","total2_kg","sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg", "trans_cost",'value_sells']]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q+'_kg']*ag5a[p]
        
(values_ag5a['total_value_p_c_nat'].isna()).sum()

for p in ['p_sell', 'p_c']:
    for area in ['_nat', '_reg', '_county',  '_district']:
        ## Using each household reported value of sells plus using p_area to value non-sold production
        values_ag5a["total2_value_"+p+area] =  values_ag5a.loc[:,['value_sells',"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5a["total2_value_"+p+area] = values_ag5a["total2_value_"+p+area].replace(0, np.nan)
        
        ## Using median selling prices to compute sold value plus using p_area to value non-sold production
        values_ag5a["total3_value_"+p+area] =  values_ag5a.loc[:,["sell_value_p_sell"+area,"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5a["total3_value_"+p+area] = values_ag5a["total3_value_"+p+area].replace(0, np.nan)
        
        
(values_ag5a[['total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat']].isna()).sum()


check = (values_ag5a[['HHID','total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat','total_value_p_c_reg','total2_value_p_c_reg','total3_value_p_c_reg']].groupby(by='HHID').sum()).replace(0,np.nan)/dollars
sum_ag5avalues = check.describe()

ag5a = values_ag5a
ag5a.reset_index(inplace=True)
ag5a.rename(columns={'index':'j'}, inplace=True)


# Merge datasets -------------------------------------------
agrica = pd.merge(agrica, ag5a, on=["HHID", 'prcid', 'pltid','cropID'], how='right')
agrica.drop_duplicates(subset=['j'],inplace=True)
agrica.set_index(['HHID','prcid','pltid','cropID'], inplace=True)
agrica = agrica.reset_index()



#for p in priceslist:
#    agrica[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]] = remove_outliers(agrica[['total_value_'+p, 'total2_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "interm_value_"+p,"gift_value_"+p]],lq=0, hq=0.98)


count_crops = pd.value_counts(agrica['cropID'])


agrica['A'] = (agrica['area_planted']).replace(0,np.nan)
agrica['y'] = (agrica['total2_value_p_c_district']).replace(0,np.nan)
agrica['season'] = 1

agrica['y'].describe()
agrica_hh = (agrica.groupby('HHID').sum()).replace(0,np.nan)
sum_agricA = agrica_hh['y'].describe(percentiles=percentiles)/dollars



#%% AGRICULTURAL SEASON 2:

#Omit rents for evaluate agricultural productivity.

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3b = pd.read_stata(my_dirct+'agsec3b.dta')
ag3b = ag3b[["HHID",'prcid', 'pltid', 'a3bq5', "a3bq7", "a3bq8", "a3bq16", "a3bq18", "a3bq19", 'a3bq30',  'a3bq28b', 'a3bq31', 'a3bq38', 'a3bq39', 'a3bq42a', 'a3bq42b','a3bq42c', 'a3bq43']]
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


ag3b = ag3b[["HHID",'prcid', 'pltid', "org_fert", "chem_fert", 'pesticides','hhlabor', 'hired_labor' , "a3bq43"]]
ag3b.columns = ["HHID",'prcid','pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']

# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
ag4b = pd.read_stata(my_dirct+'agsec4b.dta', convert_categoricals=False)
ag4b = ag4b[["HHID",'prcid', 'pltid','cropID' , 'a4bq8', 'a4bq9',  "a4bq11"]]
ag4b.columns = ["HHID",'prcid', 'pltid','itemID' , 'total_area', 'weightcrop', 'seed_cost']
ag4b = pd.merge(ag4b, crop_codes, on='itemID', how='left' )
ag4b['weightcrop'].replace(np.nan,100, inplace=True)

area = ag4b['total_area']
### already ask total area and proportion per crop
ag4b['area_planted'] = ag4b['total_area'].multiply(ag4b['weightcrop']/100, axis=0)

area_2 = ag4b['area_planted']

ag4b.reset_index(inplace=True)
ag4b.rename(columns={'index':'i'}, inplace=True)

agricb = pd.merge(ag3b, ag4b, on=['HHID','prcid','pltid'], how='right')
agricb.drop_duplicates(subset='i', inplace=True)

### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agricb['weightcrop']/100, axis=0)

# =============================================================================
# Output
# =============================================================================

ag5b = pd.read_stata(my_dirct+'agsec5b.dta', convert_categoricals=False)
ag5b = ag5b[["HHID",'prcid','pltid',"cropID","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq22"]]
ag5b.columns = ["HHID",'prcid','pltid', "itemID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]
ag5b = pd.merge(ag5b, crop_codes, on='itemID', how='left' )

# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5b.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv('C:/Users/rodri/Dropbox/JMP/data/data10/kg_conversion_10b.csv',index=False)
'''

### USE MEDIAN CONVERSION RATES ACROSS WAVES AND SEASONS
kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_10']]
kg_units['cropID'] = kg_units['cropID'].str.upper()

ag5b.replace(99999,np.nan, inplace=True)

ag5b.reset_index(inplace=True)
ag5b.rename(columns={'index':'i'}, inplace=True)
ag5b = ag5b.merge(kg_units, on=['unit','cropID'], how='left')
ag5b.drop_duplicates(subset=['i'], inplace=True) 

ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'kgconverter_10'] = ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'tokg']

ag5b['kgconverter_10'] = ag5b['kgconverter_10'].fillna(ag5b['tokg'])

# Convert to kg
ag5b[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5b["kgconverter_10"], axis="index")

for var in ['sell_kg', 'cons_kg', 'stored_kg']:
    ag5b.loc[ag5b[var]>ag5b['total_kg'], var] = ag5b.loc[ag5b[var]>ag5b['total_kg'], 'total_kg']

for var in ['gift_kg', 'food_prod_kg', 'animal_kg', 'seeds_kg']:
    ag5b.loc[ag5b[var]>ag5b['total_kg'], var] = ag5b.loc[ag5b[var]>ag5b['total_kg'], 'total_kg']


ag5b["total2_kg"] =  ag5b.loc[:,["sell_kg","gift_kg","cons_kg","food_prod_kg","animal_kg", "seeds_kg", "stored_kg"]].sum(axis=1)
ag5b["total2_kg"].replace(0,np.nan, inplace=True)


ag5b['total2_kg'] = ag5b['total2_kg'].fillna(ag5b['total_kg'])  #still many missing values 5259
ag5b['total_kg'] = ag5b['total_kg'].fillna(ag5b['total2_kg'])

check = ag5b[['total_kg', 'total2_kg']].corr()

check_ag5b_kg = (ag5b[['HHID','total_kg','total2_kg']].groupby(by='HHID').sum()).describe(percentiles)
#  0.82 correlation and almost same distribution. Good!!


# yet let's replace the few ones that reported quantities without specifying unit. Assume they reported on kg.
for var in ["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]:
    ag5b[[var+'_kg']] = ag5b[[var+'_kg']].fillna(ag5b[[var]])


#### Prices
ag5b = ag5b.merge(basic, on='HHID', how='left')
ag5b["hh_price"] = ag5b.value_sells.div(ag5b.sell_kg, axis=0) 

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
cprices_nat = pd.read_csv(folder+"pricesfood10.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood10.csv")
cprices_county = pd.read_csv(folder+"countypricesfood10.csv")
cprices_district = pd.read_csv(folder+"districtpricesfood10.csv")

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
    cprices.loc[cprices["code"]==146,"cropID"] = "SimSim"
    cprices.loc[cprices["code"]==137,"cropID"] = "Cabbage"
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
    cprices.loc[cprices["code"]== 171,"cropID"] = 'Paw Paw'
    cprices.drop(["code"], axis=1, inplace=True)
    


 # Set of prices
prices_nat = prices_nat.merge(cprices_nat, on="cropID", how="left")

#For the items that do not have consumption prices input the selling ones and viceversa.
prices_nat["m_p"] = prices_nat["m_p"].fillna(prices_nat["p_nat"])
prices_nat["p_nat"] = prices_nat["p_nat"].fillna(prices_nat["m_p"])
prices_nat.columns = ["cropID","p_sell_nat", "p_c_nat"]


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
values_ag5b = ag5b[["HHID",'prcid','pltid','cropID',"total_kg","total2_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg","trans_cost",'value_sells']]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5b[q+"_value_"+p] = ag5b[q+'_kg']*ag5b[p]
        
(values_ag5b['total_value_p_c_nat'].isna()).sum()

for p in ['p_sell', 'p_c']:
    for area in ['_nat', '_reg', '_county', '_district']:
        ## Using each household reported value of sells plus using p_area to value non-sold production
        values_ag5b["total2_value_"+p+area] =  values_ag5b.loc[:,['value_sells',"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5b["total2_value_"+p+area] = values_ag5b["total2_value_"+p+area].replace(0, np.nan)
        
        ## Using median selling prices to compute sold value plus using p_area to value non-sold production
        values_ag5b["total3_value_"+p+area] =  values_ag5b.loc[:,["sell_value_p_sell"+area,"gift_value_"+p+area,"cons_value_"+p+area,"food_prod_value_"+p+area,"animal_value_"+p+area, "stored_value_"+p+area]].sum(axis=1)
        values_ag5b["total3_value_"+p+area] = values_ag5b["total3_value_"+p+area].replace(0, np.nan)
        
        
(values_ag5b[['total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat']].isna()).sum()


check = (values_ag5b[['HHID','total_value_p_c_nat','total2_value_p_c_nat','total3_value_p_c_nat','total_value_p_c_reg','total2_value_p_c_reg','total3_value_p_c_reg']].groupby(by='HHID').sum()).replace(0,np.nan)/dollars
sum_ag5bvalues = check.describe()

ag5b = values_ag5b
ag5b.reset_index(inplace=True)
ag5b.rename(columns={'index':'j'}, inplace=True)


# Merge datasets -------------------------------------------

# Merge datasets -------------------------------------------
agricb = pd.merge(agricb, ag5b, on=["HHID",'prcid','pltid','cropID'], how='right')
agricb.drop_duplicates(subset=['j'],inplace=True)
agricb.set_index(['HHID','prcid','pltid','cropID'], inplace=True)
agricb = agricb.reset_index()



count_crops = pd.value_counts(agricb['cropID'])


agricb['A'] = (agricb['area_planted']).replace(0,np.nan)
agricb['y'] = (agricb['total2_value_p_c_district']).replace(0,np.nan)
agricb['season'] = 2

agricb['y'].describe()
agricb_hh = (agricb.groupby('HHID').sum()).replace(0,np.nan)
sum_agricB = agricb_hh['y'].describe(percentiles=percentiles)/dollars




#%%


#Big Animals------------------------------------------------------------
ag6a = pd.read_stata(my_dirct+'AGSEC6A.dta')
ag6a = ag6a[["HHID", "a6aq3", "a6aq5a", "a6aq12", "a6aq13","a6aq14","a6aq15"]]
ag6a.columns = ['HHID',"lvstid","lvstk_big" ,"bought", "value_bought", "sold", "value_sold_big"]

ag6a['p_bought'] = ag6a['value_bought']/ag6a['bought']
ag6a['p_sold'] = ag6a['value_sold_big']/ag6a['sold']

#Obtain prices animals
prices = ag6a.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6a_2010.csv')
ag6a = pd.merge(ag6a, prices, on="lvstid", how='outer')

## Big livestock wealth--------------------------------------
ag6a['lvstk_big_value'] = ag6a.lvstk_big*ag6a.p_sold_y
ag6a = ag6a.groupby(by='HHID')[["value_sold_big", 'lvstk_big_value']].sum()



#Small animals=============================================================
ag6b = pd.read_stata(my_dirct+'AGSEC6B.dta', convert_categoricals=False)
ag6b = ag6b[["HHID", "a6bq3", "a6bq5a", "a6bq12","a6bq13","a6bq14","a6bq15"]]
ag6b.columns = ['HHID', "lvstid", "lvstk_small", "bought", "value_bought", "sold", "value_sold_small"]

ag6b['p_bought'] = ag6b['value_bought']/ag6b['bought']
ag6b['p_sold'] = ag6b['value_sold_small']/ag6b['sold']


## Obtain prices --------------------------
prices = ag6b.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6b_2010.csv')
ag6b = pd.merge(ag6b, prices, on="lvstid", how='outer')

## Income: report sellings in last 6 months
ag6b['value_sold_small'] = 2*ag6b['value_sold_small'] 
## small livestock wealth--------------------------------------
ag6b['lvstk_small_value'] = ag6b.lvstk_small*ag6b.p_sold_y

ag6b = ag6b.groupby(by='HHID')[["value_sold_small", 'lvstk_small_value']].sum()



#Poultry animals==========================================
ag6c = pd.read_stata(my_dirct+'AGSEC6C.dta' )
ag6c = ag6c[["HHID", "a6cq3", "a6cq5a","a6cq12","a6cq13","a6cq14", "a6cq15"]]
ag6c.columns = ['HHID',"lvstid", "lvstk_poultry", "bought", "value_bought", "sold", "value_sold_poultry"]

ag6c['p_bought'] = ag6c['value_bought']/ag6c['bought']
ag6c['p_sold'] = ag6c['value_sold_poultry']/ag6c['sold']

## Obtain prices --------------------------
prices = ag6c.groupby(by="lvstid")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6c_2010.csv')
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
ag8.columns = ['HHID', "month_sales","month_c"]
ag8["lvst_output"] = ag8[["month_sales"]]*12
ag8["animal_c"] = ag8[["month_c"]]*12
ag8 = ag8.groupby(by='HHID').sum()
ag8.reset_index(inplace=True)



#Extension service---------------------------------------------------
ag9 = pd.read_stata(my_dirct+'agsec9.dta')
ag9 = ag9[["HHID", "a9q2","a9q9"]]
ag9.columns = ['HHID', "a9q2", "consulting_cost"]
values = [ "consulting_cost"]
index = ['HHID', "a9q2"]
panel = ag9.pivot_table(values=values, index=index)
ag9 = panel.sum(axis=0, level='HHID')



#Machinery-----------------------------------------------------------
ag10 = pd.read_stata(my_dirct+'AGSEC10.dta')
ag10 = ag10[["HHID", "itmcd","a10q2", "a10q8"]]
ag10.columns = ['HHID', "itemcd", "farm_capital","rent_tools_cost"]
ag10 = ag10.groupby(by='HHID')[["farm_capital","rent_tools_cost"]].sum()
ag10.reset_index(inplace=True)
ag10["HHID"] = pd.to_numeric(ag10['HHID'])



#Merge datasets------------------------------------------------------
livestock = pd.merge(ag6a, ag6b, on='HHID', how='outer')
livestock = pd.merge(livestock, ag6c, on='HHID', how='outer')
livestock = pd.merge(livestock, ag7, on='HHID', how='outer')
livestock = pd.merge(livestock, ag8, on='HHID', how='outer')
#livestock['HHID'] = pd.to_numeric(livestock['HHID'])



livestock["revenue_lvstk"] =livestock.loc[:,["value_sold_big","value_sold_small","value_sold_poultry","lvst_output","animal_c"]].sum(axis=1) 
livestock["revenue_lvstk"] = livestock["revenue_lvstk"].replace(0,np.nan)
livestock["cost_lvstk"] = livestock.loc[:,["animal_inp"]].sum(axis=1) 
livestock["cost_lvstk"] = livestock["cost_lvstk"].replace(0,np.nan)
livestock["wealth_lvstk"] = livestock.loc[:,["lvstk_big_value","lvstk_small_value","lvstk_poultry_value"]].sum(axis=1) 
livestock["wealth_lvstk"] = livestock["wealth_lvstk"].replace(0,np.nan)

ls = livestock[['HHID',"revenue_lvstk", "cost_lvstk", "wealth_lvstk"]]
ls = ls.dropna()

ls["HHID"] = pd.to_numeric(ls['HHID'])

# Trimming 1% 
ls["profit_lvstk"] = ls["revenue_lvstk"].fillna(0) - ls["cost_lvstk"].fillna(0)
sumls = ls.describe()/dollars




#%% Both seasons together

data = agrica.append(agricb)
data.reset_index(inplace=True)
count_crops = pd.value_counts(data['cropID'])
data['HHID'] = pd.to_numeric(data["HHID"])


### first trimming at crop level
'''
for p in priceslist:
    data[['total2_value_'+p]] = data[['cropID','total2_value_'+p]].groupby(by='cropID').apply(lambda x: remove_outliers(x, hq=0.95))
'''

#### DO TRIMMING AT HOUSEHOLD LEVEL!!!! DOESNT MAKE SENSE TO DO IT ON CROP-PLOT-HHID

#### DATA AT HOUSEHOLD LEVEL TO COMPUTE HOUSEHOLD INCOME
data_hh = data.groupby(by="HHID").sum()
data_hh = data_hh.merge(ag2a, on='HHID', how='left')
data_hh = data_hh.merge(ag2b, on='HHID', how='left')
data_hh = pd.merge(data_hh, ls, on="HHID", how="outer")
data_hh = pd.merge(data_hh, ag10, on="HHID", how="left")

data_hh.rename(columns={"HHID": "hh"}, inplace=True)

data_hh['A'] = data_hh['A']/2

data_hh['m'] = data_hh[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)

data_hh['cost_agr'] = -data_hh.loc[:,['m','rent_tools_cost']].sum(axis=1)
for p in priceslist:
    data_hh['profit_agr_'+p] = data_hh.loc[:,['total2_value_'+p,"cost_agr"]].sum(axis=1)    
    data_hh['revenue_agr_'+p] = data_hh["total2_value_"+p]
    
sumdata_hh = data_hh[['total2_value_p_c_nat','total2_value_p_c_reg', 'total2_value_p_c_county', 'total2_value_p_c_district', 
                       'total2_value_p_c_nat', 'total3_value_p_c_reg', 'total3_value_p_c_county','total3_value_p_c_district',
                       'total2_value_p_sell_nat','total2_value_p_sell_reg', 'total2_value_p_sell_county',  'total2_value_p_sell_district', 
                       'total2_value_p_sell_nat', 'total3_value_p_sell_reg', 'total3_value_p_sell_county','total3_value_p_sell_district']].describe()/dollars
  

#### agrls Wealth:
data_hh["wealth_agrls"] = data_hh.loc[:,["wealth_lvstk",'farm_capital']].sum(axis=1)
data_hh["wealth_agrls"] = data_hh["wealth_agrls"].replace(0,np.nan)

wealth = data_hh[["hh","wealth_agrls","wealth_lvstk",'farm_capital']]


### Agricultural wealth and income to compute total household W and I
wealth.to_csv(folder+"wealth_agrls10.csv", index=False)
del data_hh['wealth_agrls'], data_hh['wealth_lvstk'], data_hh['farm_capital']
data_hh.to_csv(folder+"inc_agsec10.csv", index=False)



data['m'] = (data[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)).replace(0,np.nan)
data['l'] = data[['hhlabor','hired_labor']].sum(axis=1)

#### Control for inflation =============================

basic = pd.read_stata(my_dirct+'GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","region","urban"]] 
basic['HHID'] = pd.to_numeric(basic['HHID'])
data = pd.merge(data, basic, on='HHID', how='left')


data.rename(columns={'pltid':'plotID'}, inplace=True)
data.rename(columns={'prcid':'parcelID'}, inplace=True)



data['inflation_avg'] = 0.77845969

#Divide by inflation all monetary variables: .div(data.inflation, axis=0)
data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']] = data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']].div(data.inflation_avg, axis=0)/dollars

priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 
for p in priceslist:
    data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p,"gift_value_"+p]] = data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p,"gift_value_"+p]].div(data.inflation_avg, axis=0)/dollars


data_check = data.groupby(by="HHID").sum().replace(0,np.nan)
data_check.reset_index(inplace=True)
data_check['A'] = data_check['A']/2
data_check['m'].replace(0,np.nan,inplace=True)

sumdatahh = data_check[['y','m','A','l', 'total_value_p_c_nat', 'total_value_p_c_reg', 'total_value_p_c_county', 'total_value_p_sell_nat','total_value_p_sell_reg','total_value_p_sell_county', 'total2_value_p_c_nat', 'total2_value_p_c_reg', 'total2_value_p_c_county', 'total2_value_p_sell_nat','total2_value_p_sell_reg','total2_value_p_sell_county']].describe(percentiles=percentiles)

outliers = data_check.loc[(data_check['y']>np.nanpercentile(data_check['y'], 99)) | (data_check['A']>np.nanpercentile(data_check['A'],99.9)),'HHID']

data = data[~data['HHID'].isin(outliers)]

# Check outliers land: don't seem to produce very high quantities given how much land they have. Let's consider
#them as outliers
#data_hh.loc[data_hh['hh'].isin(outliers),['total2_value_p_c_nat','total2_value_p_c_reg']]/dollars

data_check = data.groupby(by="HHID").sum()
data_check['A'] = data_check['A']/2
data_check['m'].replace(0,np.nan,inplace=True)
susumdatahh2 = data_check[['y','m','A','l', 'total_value_p_c_reg', 'total_value_p_c_county', 'total2_value_p_c_reg', 'total2_value_p_c_county','total2_value_p_c_district','total3_value_p_c_reg','total3_value_p_c_county','total3_value_p_c_district']].describe(percentiles=percentiles)

data.to_csv(folder+'agric_data10.csv', index=False)
print('data agric 2010 saved')