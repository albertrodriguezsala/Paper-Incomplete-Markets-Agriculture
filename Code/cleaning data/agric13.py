# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:17:10 2018

@author: Albert
"""

# =============================================================================
# #### Agricultural data Uganda 2010-11
# =============================================================================


'''
DESCRIPTION:
    Uses the data from the 2013-2014 Integrated Survey in Agriculture from the UNPS (ISA-LSMS) to obtain:
        For the 2 main harvest of the year of the survey. (fall 2013 and spring 2014/)
            -agricultural inputs variables at plot level and household level.
            -agricultural ouptut variables at plot level, household level, and crop level.   
        -household land, household farming assets (capital) household level
        - Livestock revenues (including non-sold consumption), costs, and stock (wealth) household level
    output: inc_agsec13.csv (agric income and costs at hh level) wealth_agrls13.csv (agric wealth: livestock+farming capital), agric_data13.csv (outputs and inputs at crop-plot-hh level)
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

my_dirct = str(dirct)+'/data/raw data/2013/'  
folder =  str(dirct)+'/data/data13/'
folder2 = str(dirct)+'/data/auxiliary data/'

pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89

hq=0.995
hq_2= 0.95
lq=0.0

percentiles = [ 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

# Id problem
ag1 = pd.read_csv(my_dirct+'agsec1.csv', header=0, na_values='NA')
ag1= ag1[["hh","HHID"]]

basic = pd.read_csv(my_dirct+'gsec1.csv', header=0, na_values='NA')
basic = basic[["HHID", "HHID_old","region","urban","year", "month","sregion", 'h1aq1a',  'h1aq3b', 'h1aq4b']] 
basic.columns = ["hh", "hh_old","region","urban","year", "month","sregion", 'district_code', 'subcounty', 'parish']
district_data = pd.read_csv(folder2+'district_codename.csv')
basic = basic.merge(district_data, on='district_code')
basic['subcounty'] = basic['subcounty'].str.upper()
### I lose 1200 obs with merging with subcounty 2011
county = pd.read_csv(folder2+'county_subcounty.csv')
basic = basic.merge(county, on='subcounty', how='left')

blu = basic['district'].value_counts()

basic = basic.merge(ag1,on='hh', how='right')
#del basic['hh']



#%% AGRICULTURAL SEASON 1:


#rent obtained ========================================================
ag2a = pd.read_csv(my_dirct+'agsec2a.csv', header=0, na_values='NA')
ag2a = ag2a[["HHID", "a2aq14"]]
ag2a = ag2a.groupby(by="HHID")[["a2aq14"]].sum()
ag2a.columns = ["rent_owner"]
ag2a.reset_index(inplace=True)
ag2a["HHID"] = pd.to_numeric(ag2a['HHID'])


# rent payment ========================================================
ag2b = pd.read_csv(my_dirct+'agsec2b.csv', header=0, na_values='NA')
ag2b = ag2b[["HHID", "a2bq9", "a2bq13"]]
values = ["a2bq9", "a2bq13"]
ag2b = ag2b.groupby(by="HHID")[["a2bq9", "a2bq13"]].sum()

ag2b["rent_noowner"] = ag2b["a2bq13"].fillna(0) - ag2b["a2bq9"].fillna(0)
ag2b["rent_noowner"] = ag2b["rent_noowner"].replace(0, np.nan)
ag2b = ag2b[["rent_noowner"]]
ag2b.reset_index(inplace=True)
ag2b["HHID"] = pd.to_numeric(ag2b['HHID'])
#rent obtained - payed for those who rend

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3a = pd.read_csv(my_dirct+'agsec3a.csv', header=0, na_values='NA')
ag3a = ag3a[["HHID", 'parcelID','plotID', 'a3aq5', 'a3aq7', "a3aq8", 'a3aq15', 'a3aq17', "a3aq18",'a3aq24a','a3aq24b', "a3aq27", 'a3aq33a_1', 'a3aq33b_1', 'a3aq33c_1', 'a3aq33d_1', 'a3aq33e_1', 'a3aq35a', 'a3aq35b','a3aq35c' ,"a3aq36"]]
ag3a['hhlabor'] = ag3a["a3aq33a_1"].fillna(0) +ag3a["a3aq33b_1"].fillna(0) +ag3a["a3aq33c_1"].fillna(0) +ag3a["a3aq33d_1"].fillna(0)+ ag3a["a3aq33e_1"].fillna(0)
ag3a['hired_labor'] = ag3a["a3aq35a"].fillna(0) +ag3a["a3aq35b"].fillna(0)   +ag3a["a3aq35c"].fillna(0)

ag3a['p_orgfert'] = np.nanmedian( ag3a['a3aq8'].div(ag3a['a3aq7'], axis=0) )
ag3a['p_chemfert'] = np.nanmedian( ag3a['a3aq18'].div(ag3a['a3aq17'], axis=0) )
#ag3a['p_pest'] = np.nanmedian( ag3a['a3aq27'].div(ag3a['a3aq26'], axis=0) )  ## missing quantity pesticies purchased in the data (but in the questionnaire is there)

ag3a['org_fert'] = ag3a['p_orgfert']*ag3a['a3aq5']
### important changes when included non-bought 
ag3a['org_fert'].describe()
ag3a['a3aq8'].describe()

## with chemfert might not be important
ag3a['chem_fert'] = ag3a['p_chemfert']*ag3a['a3aq17']
ag3a['chem_fert'].describe()
ag3a['a3aq18'].describe()


ag3a = ag3a[["HHID", 'plotID', "org_fert", "chem_fert", "a3aq27",'hhlabor', 'hired_labor', "a3aq36"]]
ag3a.columns = ["HHID", 'plotID','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4a = pd.read_csv(my_dirct+'agsec4a.csv', header=0, na_values='NA')
ag4a = ag4a[["HHID", 'parcelID','plotID', 'cropID' , 'a4aq7', 'a4aq9', "a4aq15", 'a4aq13']]
ag4a = ag4a[["HHID", 'plotID', 'cropID', 'a4aq7', 'a4aq9',  "a4aq15"]]
ag4a.columns = ["HHID", 'plotID','itemID' , 'total_area', 'weightcrop', 'seed_cost']
ag4a['weightcrop'].replace(np.nan,100, inplace=True)

crop_codes = pd.read_csv(folder2+'crop_codes.csv')
crop_codes.columns = ['itemID','cropID']
ag4a = pd.merge(ag4a, crop_codes, on='itemID', how='left' )

### already ask total area and proportion per crop
ag4a['area_planted'] = ag4a['total_area'].multiply(ag4a['weightcrop']/100, axis=0)


ag4a.reset_index(inplace=True)
ag4a.rename(columns={'index':'i'}, inplace=True)
agrica = pd.merge(ag3a, ag4a, on=['HHID','plotID'], how='right')
agrica.drop_duplicates(subset=['i'], inplace=True) 

### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agrica[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agrica['weightcrop']/100, axis=0)



# =============================================================================
# Output
# =============================================================================

ag5a = pd.read_csv(my_dirct+'agsec5a.csv', header=0, na_values='NA')
ag5a = ag5a[["HHID",'plotID',"cropID","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq7d","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq21"]]
ag5a.columns = ["HHID", 'plotID', "itemID", "total","unit", "tokg", "sell", "unit2","tokg2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]
ag5a.loc[ag5a.unit==1, "tokg"] = 1
ag5a = pd.merge(ag5a, crop_codes, on='itemID', how='left' )

lele = ag5a['cropID'].value_counts()


# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5a.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv(folder+'/kg conversion/kg_conversion_13a.csv', index=False)
'''
### USE MEDIAN CONVERSION RATES ACROSS WAVES AND SEASONS
kg_units = pd.read_csv(folder2+'/kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_13']]
ag5a.replace(99999,np.nan, inplace=True)

ag5a.reset_index(inplace=True)
ag5a.rename(columns={'index':'i'}, inplace=True)
ag5a = ag5a.merge(kg_units, on=['unit','cropID'], how='left')
ag5a.drop_duplicates(subset=['i'], inplace=True) 

ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'kgconverter_13'] = ag5a.loc[(ag5a['unit']==99)|(ag5a['unit']==87)|(ag5a['unit']==80),'tokg']

ag5a['kgconverter_13'] = ag5a['kgconverter_13'].fillna(ag5a['tokg'])

# Convert to kg
ag5a[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5a["kgconverter_13"], axis="index")

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


# yet let's replace the few ones that reported quantities without specifying unit. Assume they reported on kg.
for var in ["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]:
    ag5a[[var+'_kg']] = ag5a[[var+'_kg']].fillna(ag5a[[var]])


#### Prices
ag5a = ag5a.merge(basic, on='HHID', how='left')
ag5a["hh_price"] = ag5a.value_sells.div(ag5a.sell_kg, axis=0) 

# Set ipython's max row display
ag5a[['hh_price']].describe()

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

prices_nat.describe()

# save prices to use in 2015
prices_nat.to_csv(folder+'nat_prices13a.csv')
prices_reg.to_csv(folder+'reg_prices13a.csv')
prices_county.to_csv(folder+'county_prices13a.csv')
prices_district.to_csv(folder+'district_prices13a.csv')

## Use consumption prices
cprices_nat = pd.read_csv(folder+"pricesfood13.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood13.csv")
cprices_county = pd.read_csv(folder+"countypricesfood13.csv")
cprices_district = pd.read_csv(folder+"districtpricesfood13.csv")

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

## add reasonable prices for those missing
prices_nat.reset_index(inplace=True)
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Chick Peas', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Pigeon Peas', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Chick Peas', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Pigeon Peas', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Carrots', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Sweet Potatoes', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Carrots', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Sweet Potatoes', ["p_sell_nat"]].iloc[0,0]


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
values_ag5a = ag5a[["HHID", 'plotID', 'cropID',"total_kg","total2_kg","sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg", "trans_cost",'value_sells']]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q+'_kg']*ag5a[p]
        
(values_ag5a['total_value_p_c_nat'].isna()).sum()

for p in ['p_sell', 'p_c']:
    for area in ['_nat', '_reg', '_county','_district']:
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
agrica = pd.merge(agrica, ag5a, on=["HHID",  'plotID','cropID'], how='right')
agrica.drop_duplicates(subset=['j'],inplace=True)
agrica.set_index(['HHID','plotID','cropID'], inplace=True)
agrica = agrica.reset_index()



count_crops = pd.value_counts(agrica['cropID'])


agrica['A'] = (agrica['area_planted']).replace(0,np.nan)
agrica['y'] = (agrica['total3_value_p_c_district']).replace(0,np.nan)
agrica['season'] = 1

agrica['y'].describe()
agrica_hh = (agrica.groupby('HHID').sum()).replace(0,np.nan)
sum_agricA = agrica_hh['y'].describe(percentiles=percentiles)/dollars



#%% AGRICULTURAL SEASON 2:

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3b = pd.read_csv(my_dirct+'agsec3b.csv', header=0, na_values='NA')
ag3b = ag3b[["HHID", 'parcelID','plotID', 'a3bq5', 'a3bq7', "a3bq8", 'a3bq15', 'a3bq17', "a3bq18",'a3bq24a','a3bq24b', 'a3bq26',"a3bq27", 'a3bq33a_1', 'a3bq33b_1', 'a3bq33c_1', 'a3bq33d_1', 'a3bq33e_1', 'a3bq35a', 'a3bq35b','a3bq35c' ,"a3bq36"]]
ag3b['hhlaborb'] = ag3b["a3bq33a_1"].fillna(0) +ag3b["a3bq33b_1"].fillna(0) +ag3b["a3bq33c_1"].fillna(0) +ag3b["a3bq33d_1"].fillna(0)+ ag3b["a3bq33e_1"].fillna(0)
ag3b['hired_laborb'] = ag3b["a3bq35a"].fillna(0) +ag3b["a3bq35b"].fillna(0)   +ag3b["a3bq35c"].fillna(0)


ag3b['p_orgfert'] = np.nanmedian( ag3b['a3bq8'].div(ag3b['a3bq7'], axis=0) )
ag3b['p_chemfert'] = np.nanmedian( ag3b['a3bq18'].div(ag3b['a3bq17'], axis=0) )
ag3b['p_pest'] = np.nanmedian( ag3b['a3bq27'].div(ag3b['a3bq26'], axis=0) )  ## missing quantity pesticies purchased in the data (but in the questionnaire is there)

ag3b['org_fert'] = ag3b['p_orgfert']*ag3b['a3bq5']
### important changes when included non-bought 
ag3b['org_fert'].describe()
ag3b['a3bq8'].describe()

## with chemfert might not be important: indeed same number of obs.
ag3b['pesticides'] = ag3b['p_pest']*ag3b['a3bq24b']
ag3b['pesticides'].describe()
ag3b['a3bq27'].describe()

##pesticides: small changes
ag3b['chem_fert'] = ag3b['p_chemfert']*ag3b['a3bq17']
ag3b['chem_fert'].describe()
ag3b['a3bq18'].describe()


ag3b = ag3b[["HHID", 'plotID', "org_fert", "chem_fert", 'pesticides','hhlaborb', 'hired_laborb', "a3bq36"]]
ag3b.columns = ["HHID", 'plotID','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4b = pd.read_csv(my_dirct+'agsec4b.csv', header=0, na_values='NA')
ag4b = ag4b[["HHID", 'plotID', 'cropID', 'a4bq7', 'a4bq9', "a4bq15"]]
ag4b = ag4b[["HHID", 'plotID' , 'cropID', 'a4bq7', 'a4bq9',  "a4bq15"]]
ag4b.columns = ["HHID", 'plotID','itemID' , 'total_area', 'weightcrop', 'seed_cost']
ag4b['weightcrop'].replace(np.nan,100, inplace=True)

crop_codes = pd.read_csv(folder2+'crop_codes.csv')
crop_codes.columns = ['itemID','cropID']
ag4b = pd.merge(ag4b, crop_codes, on='itemID', how='left' )

### already ask total area and proportion per crop
ag4b['area_planted'] = ag4b['total_area'].multiply(ag4b['weightcrop']/100, axis=0)

ag4b.reset_index(inplace=True)
ag4b.rename(columns={'index':'i'}, inplace=True)

agricb = pd.merge(ag3b, ag4b, on=['HHID','plotID'], how='right')
agricb.drop_duplicates(subset='i', inplace=True)

### assign weight corresponding to its crop land size. In other words, assume labor and intermediates are split it equally among plot.
agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']] = agricb[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']].multiply(agricb['weightcrop']/100, axis=0)




# =============================================================================
# Output
# =============================================================================

ag5b = pd.read_csv(my_dirct+'agsec5b.csv', header=0, na_values='NA')
ag5b = ag5b[["HHID",'plotID',"cropID","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq7d","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq21"]]
ag5b.columns = ["HHID", 'plotID', "itemID", "total","unit", "tokg", "sell", "unit2","tokg2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]

ag5b = pd.merge(ag5b, crop_codes, on='itemID', how='left' )

lele = ag5b['cropID'].value_counts()


# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
'''
conversion_kg = ag5b.groupby(by=["unit",'cropID'])[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit",'cropID',"kgconverter"]
conversion_kg.to_csv(folder2+'kg conversion/kg_conversion_13b.csv', index=False)
'''

kg_units = pd.read_csv(folder2+'kg conversion/conversionkg_allwaves.csv')
kg_units = kg_units[['unit','cropID','kgconverter_13']]

ag5b.replace(99999,np.nan, inplace=True)

ag5b.reset_index(inplace=True)
ag5b.rename(columns={'index':'i'}, inplace=True)
ag5b = ag5b.merge(kg_units, on=['unit','cropID'], how='left')
ag5b.drop_duplicates(subset=['i'], inplace=True) 

ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'kgconverter_13'] = ag5b.loc[(ag5b['unit']==99)|(ag5b['unit']==87)|(ag5b['unit']==80),'tokg']

ag5b['kgconverter_13'] = ag5b['kgconverter_13'].fillna(ag5b['tokg'])

# Convert to kg
ag5b[["total_kg", "sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg"]] = ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5b["kgconverter_13"], axis="index")

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



# save prices to use in 2015
prices_nat.to_csv(folder+'nat_prices13b.csv')
prices_reg.to_csv(folder+'reg_prices13b.csv')
prices_county.to_csv(folder+'county_prices13b.csv')
prices_district.to_csv(folder+'district_prices13b.csv')

## Use consumption prices
cprices_nat = pd.read_csv(folder+"pricesfood13.csv")
cprices_reg = pd.read_csv(folder+"regionpricesfood13.csv")
cprices_county = pd.read_csv(folder+"countypricesfood13.csv")
cprices_district = pd.read_csv(folder+"districtpricesfood13.csv")

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

## add reasonable prices for those missing
prices_nat.reset_index(inplace=True)
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Coco Yam', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Yam', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Chick Peas', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Pigeon Peas', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Chick Peas', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Pigeon Peas', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Wheat', ["p_c_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_c_nat"]].iloc[0,0]
prices_nat.loc[prices_nat['cropID']=='Wheat', ["p_sell_nat"]] = prices_nat.loc[prices_nat['cropID']=='Finger Millet', ["p_sell_nat"]].iloc[0,0]

prices_nat.loc[prices_nat['cropID']=='Coffee All', ["p_c_nat"]] = 1111.11  #previous season price
prices_nat.loc[prices_nat['cropID']=='Coffee All', ["p_sell_nat"]] = 1111.11

prices_nat.loc[prices_nat['cropID']=='Cow Peas', ["p_c_nat"]] = 1000  #previous season price
prices_nat.loc[prices_nat['cropID']=='Cow Peas', ["p_sell_nat"]] = 1000

prices_nat.loc[prices_nat['cropID']=='Pawpaw', ["p_c_nat"]] = 250  #previous season price
prices_nat.loc[prices_nat['cropID']=='Pawpaw', ["p_sell_nat"]] = 250

prices_nat.loc[prices_nat['cropID']=='Vanilla', ["p_c_nat"]] = 7500  #previous season price
prices_nat.loc[prices_nat['cropID']=='Vanilla', ["p_sell_nat"]] = 7500


prices_nat.loc[prices_nat['cropID']=='Yam', ["p_c_nat"]] = 651.85  #previous season price
prices_nat.loc[prices_nat['cropID']=='Yam', ["p_sell_nat"]] = 651.85



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
values_ag5b = ag5b[["HHID", 'plotID', 'cropID',"total_kg","total2_kg","sell_kg", "gift_kg", "cons_kg", "food_prod_kg", "animal_kg", "seeds_kg", "stored_kg", "trans_cost",'value_sells']]
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
agricb = pd.merge(agricb, ag5b, on=["HHID",  'plotID','cropID'], how='right')
agricb.drop_duplicates(subset=['j'],inplace=True)
agricb.set_index(['HHID','plotID','cropID'], inplace=True)
agricb = agricb.reset_index()

count_crops = pd.value_counts(agricb['cropID'])

agricb['A'] = (agricb['area_planted']).replace(0,np.nan)
agricb['y'] = (agricb['total3_value_p_c_district']).replace(0,np.nan)
agricb['season'] = 2

agricb['y'].describe()
agricb_hh = (agricb.groupby('HHID').sum()).replace(0,np.nan)
sum_agricB = agricb_hh['y'].describe(percentiles=percentiles)/dollars

#%% Livestock

#Big Animals=======================================================
ag6a = pd.read_csv(my_dirct+'agsec6a.csv', header=0, na_values='NA')
ag6a = ag6a[["hh", "LiveStockID", "a6aq3a", "a6aq5c", "a6aq13a", "a6aq13b","a6aq14a","a6aq14b"]]
ag6a.columns = ['hh',"LiveStockID","lvstk_big" ,"labour_big","bought", "p_bought", "sold", "p_sold"]

#Obtain prices animals
prices = ag6a.groupby(by="LiveStockID")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6a_2013.csv')

ag6a = pd.merge(ag6a, prices, on="LiveStockID", how='outer')

## Income selling big livestock----------------
ag6a["value_bought"] = ag6a.bought*ag6a.p_bought_y
ag6a["value_sold_big"] = ag6a.sold*ag6a.p_sold_y

## Big livestock wealth--------------------------------------
ag6a['lvstk_big_value'] = ag6a.lvstk_big*ag6a.p_sold_y
ag6a = ag6a.groupby(by='hh')[["labour_big","value_sold_big", 'lvstk_big_value']].sum()

#Small animals=============================================================
ag6b = pd.read_csv(my_dirct+'agsec6b.csv', header=0, na_values='NA')
ag6b = ag6b[["hh", "ALiveStock_Small_ID", "a6bq3a", "a6bq5c","a6bq13a","a6bq13b","a6bq14a","a6bq14b"]]
ag6b.columns = ["hh", "ALiveStock_Small_ID", "lvstk_small", "labour_small", "bought", "p_bought", "sold", "p_sold"]

## Obtain prices --------------------------
prices = ag6b.groupby(by="ALiveStock_Small_ID")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6b_2013.csv')

ag6b = pd.merge(ag6b, prices, on="ALiveStock_Small_ID", how='outer')

# Income selling small livestock-----------------------
ag6b["value_bought"] = ag6b.bought*ag6b.p_bought_y
ag6b["value_sold_small"] = ag6b.sold*ag6b.p_sold_y

## small livestock wealth--------------------------------------
ag6b['lvstk_small_value'] = ag6b.lvstk_small*ag6b.p_sold_y

ag6b = ag6b.groupby(by='hh')[["labour_small","value_sold_small", 'lvstk_small_value']].sum()


#Poultry animals==========================================
ag6c = pd.read_csv(my_dirct+'agsec6c.csv', header=0, na_values='NA')
ag6c = ag6c[["hh", "APCode", "a6cq3a", "a6cq5c","a6cq13a","a6cq13b","a6cq14a","a6cq14b"]]
ag6c.columns = ['hh',"APCode", "lvstk_poultry" ,"labour_small2","bought", "p_bought", "sold", "p_sold"]

## Obtain prices --------------------------
prices = ag6c.groupby(by="APCode")[["p_bought","p_sold"]].median()
p= prices.mean()
prices = prices.fillna(p)
prices.to_csv(folder+'prices_6c_2013.csv')

ag6c = pd.merge(ag6c, prices, on="APCode", how='outer')

# Income selling poultry livestock-----------------------
# Sellings poultry in last 3 months
ag6c["value_bought"] = 4*ag6c.bought*ag6c.p_bought_y
ag6c["value_sold_poultry"] = ag6c.sold*ag6c.p_sold_y

## poultry livestock wealth--------------------------------------
ag6c['lvstk_poultry_value'] = ag6c.lvstk_poultry*ag6c.p_sold_y
ag6c = ag6c.groupby(by='hh')[["labour_small2","value_sold_poultry", 'lvstk_poultry_value']].sum()


# Livestock inputs===========================================
ag7 = pd.read_csv(my_dirct+'agsec7.csv', header=0, na_values='NA')
ag7 = ag7[["hh", "AGroup_ID","a7bq2e", "a7bq3f", "a7bq5d","a7bq6c","a7bq7c", "a7bq8c"]]

ag7 = ag7.groupby(by="hh").sum()
ag7["animal_inp"] = ag7.loc[:,["a7bq2e","a7bq3f","a7bq5d","a7bq6c","a7bq7c","a7bq8c"]].sum(axis=1)
ag7 = ag7[["animal_inp"]]
ag7 = ag7.replace(0,np.nan)
ag7.reset_index(inplace=True)
ag7.columns = ['hh','animal_inp']
#COST


# Livestock Outputs================================================

#Meat sold-----------------------------------------------------------
ag8a = pd.read_csv(my_dirct+'agsec8a.csv', header=0, na_values='NA')
ag8a = ag8a[["hh", "a8aq5"]]
ag8a = ag8a.groupby(by='hh').sum()
ag8a.reset_index(inplace=True)
ag8a.columns = ['hh',"meat_sold"]


# Milk  Sold------------------------------------------------
ag8b = pd.read_csv(my_dirct+'agsec8b.csv', header=0, na_values='NA')
ag8b = ag8b[["hh", "a8bq9"]]
ag8b.columns = ["hh",  "milk_sold"]
ag8b = ag8b.groupby(by='hh').sum()



# Eggs--------------------------------------------
ag8c = pd.read_csv(my_dirct+'agsec8c.csv', header=0, na_values='NA')
ag8c = ag8c[["hh", "a8cq5"]]
ag8c.columns = ["hh", "egg_sold"]
ag8c.egg_sold = 4*ag8c.egg_sold
ag8c = ag8c.groupby(by='hh').sum()



#Machinery and farm capital--------------------------------------------------
ag10 = pd.read_csv(my_dirct+'agsec10.csv', header=0, na_values='NA')
ag10 = ag10[["HHID", 'a10q2',"a10q8"]]
ag10.columns = ["HHID", "farm_capital", "rent_tools_cost"]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()



#Merge datasets------------------------------------------------------

livestock = pd.merge(ag6a, ag6b, on='hh', how='outer')
livestock = pd.merge(livestock, ag6c, on='hh', how='outer')
livestock = pd.merge(livestock, ag7, on='hh', how='outer')
livestock = pd.merge(livestock, ag8a, on='hh', how='outer')
livestock = pd.merge(livestock, ag8b, on='hh', how='outer')
livestock = pd.merge(livestock, ag8c, on='hh', how='outer')


del ag6c, ag7,ag8a,ag8b, prices, p,
#Pass it to dollars to see if values make sense or not
livestock2 = livestock.loc[:, livestock.columns != 'hh']/2586.89

summaryl1 = livestock2.describe()

# Self-consumed production recovered by consumption questionaire:
animal_c = pd.read_csv(folder+"c_animal13.csv")

livestock = pd.merge(livestock, animal_c, on="hh", how="outer")
livestock.rename(columns={'own_value':'animal_c'}, inplace=True)


livestock["revenue_lvstk"] =livestock.loc[:,["value_sold_big","value_sold_small","value_sold_poultry","meat_sold","animal_c","milk_sold", "egg_sold"]].sum(axis=1) 
livestock["revenue_lvstk"] = livestock["revenue_lvstk"].replace(0,np.nan)
livestock["cost_lvstk"] = livestock.loc[:,["animal_inp"]].sum(axis=1) 
livestock["cost_lvstk"] = livestock["cost_lvstk"].replace(0,np.nan)
livestock["wealth_lvstk"] = livestock.loc[:,["lvstk_big_value","lvstk_small_value","lvstk_poultry_value"]].sum(axis=1) 
livestock["wealth_lvstk"] = livestock["wealth_lvstk"].replace(0,np.nan)

ls = livestock[["hh","revenue_lvstk", "cost_lvstk", "wealth_lvstk"]]
ls = ls.dropna()


# Trimming 1% 
ls["profit_lvstk"] = ls["revenue_lvstk"].fillna(0) - ls["cost_lvstk"].fillna(0)


sumls = ls.describe()/dollars




#%% Both seasons together

data = agrica.append(agricb)
data.reset_index(inplace=True)

# Id problem
ag1 = pd.read_csv(my_dirct+'agsec1.csv', header=0, na_values='NA')
ag1= ag1[["hh","HHID"]] 

#### DATA AT HOUSEHOLD LEVEL TO COMPUTE HOUSEHOLD INCOME
data_hh = data.groupby(by="HHID").sum()
data_hh = data_hh.merge(ag2a, on='HHID', how='left')
data_hh = data_hh.merge(ag2b, on='HHID', how='left')
data_hh = pd.merge(data_hh, ag1, on="HHID", how="outer")
data_hh = pd.merge(data_hh, ls, on="hh", how="left")
data_hh = pd.merge(data_hh, ag10, on="HHID", how="left")

#data_hh = data_hh.merge(ag1, on='HHID', how='right') ### already merged with livestock above.

data_hh['A'] = data_hh['A']/2


data_hh['m'] = data_hh[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)
data_hh['l'] = data_hh[['hhlabor','hired_labor']].sum(axis=1)

data = pd.merge(data, ag1, on='HHID', how='left')
del data['HHID']
data.rename(columns={'hh':'HHID'}, inplace=True)

data_hh['m'] = data_hh[['org_fert', 'chem_fert', 'pesticides','seed_cost']].sum(axis=1)

data_hh['cost_agr'] = -data_hh.loc[:,['m','rent_tools_cost']].sum(axis=1)
for p in priceslist:
    data_hh['profit_agr_'+p] = data_hh.loc[:,['total3_value_'+p,"cost_agr"]].sum(axis=1)    
    data_hh['revenue_agr_'+p] = data_hh["total3_value_"+p]
    
sumdata_hh = data_hh[['total2_value_p_c_nat','total2_value_p_c_reg', 'total2_value_p_c_county', 
                       'total2_value_p_c_nat', 'total3_value_p_c_reg', 'total3_value_p_c_county',
                       'total2_value_p_sell_nat','total2_value_p_sell_reg', 'total2_value_p_sell_county', 
                       'total2_value_p_sell_nat', 'total3_value_p_sell_reg', 'total3_value_p_sell_county']].describe()/dollars
      


#### agrls Wealth:
data_hh["wealth_agrls"] = data_hh.loc[:,["wealth_lvstk",'farm_capital']].sum(axis=1)
data_hh["wealth_agrls"] = data_hh["wealth_agrls"].replace(0,np.nan)

wealth = data_hh[['hh',"HHID","wealth_agrls","wealth_lvstk",'farm_capital']]


### Agricultural wealth and income to compute total household W and I
wealth.to_csv(folder+"wealth_agrls13.csv", index=False)
del data_hh['wealth_agrls'], data_hh['wealth_lvstk'], data_hh['farm_capital']
data_hh.to_csv(folder+"inc_agsec13.csv", index=False)


## DATA AT HHID-PLOT-CROP-SEASON LEVEL TO CROP PRODUCTIVITY STUDY:
basic = pd.read_csv(my_dirct+'gsec1.csv', header=0, na_values='NA')
basic = basic[["HHID","region","urban"]] 
data = pd.merge(data, basic, on='HHID', how='left')



data['m'] = data[['org_fert', 'chem_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].sum(axis=1)
data['l'] = data[['hhlabor','hired_labor']].sum(axis=1)

data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']] = data[['chem_fert',  'm', 'org_fert', 'pesticides', 'seed_cost',  'trans_cost', 'y', 'labor_payment']]/dollars

priceslist = ["p_sell_nat", "p_c_nat","p_sell_reg", "p_c_reg","p_sell_county", "p_c_county","p_sell_district", "p_c_district"] 
for p in priceslist:
    data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "gift_value_"+p]] = data[['total_value_'+p, 'total2_value_'+p, 'total3_value_'+p, "sell_value_"+p, "cons_value_"+p, "stored_value_"+p, "gift_value_"+p]]/dollars


data_check = data.groupby(by="HHID").sum()
data_check.reset_index(inplace=True)
data_check['A'] = data_check['A']/2
sumdatahh = data_check[['y','total_value_p_c_nat', 'total_value_p_c_reg', 'total_value_p_c_county', 'total_value_p_sell_nat','total_value_p_sell_reg','total_value_p_sell_county',
                     'total2_value_p_c_nat', 'total2_value_p_c_reg', 'total2_value_p_c_county', 'total2_value_p_sell_nat','total2_value_p_sell_reg','total2_value_p_sell_county']].describe()

outliers = data_check.loc[(data_check['y']>np.percentile(data_check['y'], 99)) | (data_check['m']>np.percentile(data_check['m'], 100)) |(data_check['A']>np.percentile(data_check['A'],99.9)),'HHID']

data = data[~data['HHID'].isin(outliers)]

data_check = data.groupby(by="HHID").sum()
data_check.reset_index(inplace=True)
data_check['A'] = data_check['A']/2


sumdatahh2 = data_check[['y','m','A','l', 'total_value_p_c_reg', 'total_value_p_c_county', 'total2_value_p_c_reg', 'total2_value_p_c_county', 'total2_value_p_c_district','total3_value_p_c_reg', 'total3_value_p_c_county',  'total3_value_p_c_district']].describe(percentiles=percentiles)

print(sumdatahh2)
data.to_csv(folder+'agric_data13.csv', index=False)
print('data agric 13 saved')


