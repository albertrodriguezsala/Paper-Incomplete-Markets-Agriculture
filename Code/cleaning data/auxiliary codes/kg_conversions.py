# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:28:05 2021

@author: rodri
"""


# =============================================================================
#  Κg conversion rates of the units across waves: output and consumption
# =============================================================================


import pandas as pd
import os
os.chdir('C:/Users/rodri/Dropbox/JMP/data/auxiliary data/kg conversion/')


## 2009
ag09_a = pd.read_csv('kg_conversion_09a.csv')
ag09_a.columns = ['unit','cropID','kgconverter_a']
ag09_b = pd.read_csv('kg_conversion_09b.csv')
ag09_b.columns = ['unit','cropID', 'kgconverter_b']
ag09 = ag09_a.merge(ag09_b, on=['unit','cropID'], how='outer')
ag09['kgconverter_09'] = ag09[['kgconverter_a','kgconverter_b']].mean(axis=1, skipna=True)
ag09.dropna(subset=['kgconverter_09'], inplace=True)

## 2010
ag10_a = pd.read_csv('kg_conversion_10a.csv')
ag10_a.columns = ['unit','cropID','kgconverter_a']
ag10_b = pd.read_csv('kg_conversion_10b.csv')
ag10_b.columns = ['unit','cropID' ,'kgconverter_b']
ag10 = ag10_a.merge(ag10_b, on=['unit','cropID'], how='outer')
ag10['cropID'] = ag10['cropID'].str.title()
ag10['kgconverter_10'] = ag10[['kgconverter_a','kgconverter_b']].mean(axis=1, skipna=True)
ag10.dropna(subset=['kgconverter_10'], inplace=True)

## 2011
# 2011 Simsim
ag11_codes = pd.read_csv('kg_conversion_11a.csv')
ag11_codes = ag11_codes[['unit','unit_name']]
ag11_a =  pd.read_csv('kg_conversion_11a_2.csv')
ag11_a.columns = ['unit', 'cropID','kgconverter_a']


ag11_codes = pd.read_csv('kg_conversion_11b.csv')
ag11_codes = ag11_codes[['unit','unit_name']]
ag11_b =  pd.read_csv('kg_conversion_11b_2.csv')
ag11_b.columns = ['unit', 'cropID','kgconverter_b']


ag11 = ag11_a.merge(ag11_b, on=['unit','cropID'], how='outer')
ag11['kgconverter_11'] = ag11[['kgconverter_a','kgconverter_b']].mean(axis=1, skipna=True)
ag11.dropna(subset=['kgconverter_11'], inplace=True)
ag11['cropID'] = ag11['cropID'].str.title()

## 2013
ag13_a = pd.read_csv('kg_conversion_13a.csv')
ag13_a.columns = ['unit','cropID','kgconverter_a']
ag13_b = pd.read_csv('kg_conversion_13b.csv')
ag13_b.columns = ['unit','cropID','kgconverter_b']
ag13 = ag13_a.merge(ag13_b, on=['unit','cropID'], how='outer')
ag13['kgconverter_13'] = ag13[['kgconverter_a','kgconverter_b']].mean(axis=1, skipna=True)
ag13.dropna(subset=['kgconverter_13'], inplace=True)

## 2015
ag15_a = pd.read_csv('kg_conversion_15a.csv')
ag15_a.columns = ['unit','cropID','kgconverter_a']
ag15_b = pd.read_csv('kg_conversion_15b.csv')
ag15_b.columns = ['unit','cropID','kgconverter_b']
ag15 = ag15_a.merge(ag15_b, on=['unit','cropID'], how='outer')
ag15['kgconverter_15'] = ag15[['kgconverter_a','kgconverter_b']].mean(axis=1, skipna=True)
ag15.dropna(subset=['kgconverter_15'], inplace=True)


kg_allwaves = pd.merge(ag09[['unit','cropID','kgconverter_09']],ag10[['unit','cropID','kgconverter_10']], on=['unit','cropID'], how='outer')
kg_allwaves = pd.merge(kg_allwaves ,ag11[['unit','cropID','kgconverter_11']], on=['unit','cropID'], how='outer')
kg_allwaves = pd.merge(kg_allwaves ,ag13[['unit','cropID','kgconverter_13']], on=['unit','cropID'], how='outer')
kg_allwaves = pd.merge(kg_allwaves ,ag15[['unit','cropID','kgconverter_15']], on=['unit','cropID'], how='outer')

kg_allwaves.sort_values(by='unit', inplace=True)

# I manually look at outliers --------------

# First some code units must be wrong (not in codebook, or being like bottles, fish, grams that their conversions dont make sense)
wrong_units = [0,2,3,4,5,6,7,8,23,24,25,27,35,36,41,42,45,48,49,55,58,59,60,61,62,86]
kg_allwaves = kg_allwaves[~kg_allwaves['unit'].isin(wrong_units)]

kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==11) & (kg_allwaves['cropID']=='Sun Flower')].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==12) & (kg_allwaves['cropID']=='Cotton')].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==12) & (kg_allwaves['cropID']=='Sun Flower') ].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==32) & (kg_allwaves['cropID']=='Green Gram') ].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==83) & (kg_allwaves['cropID']=='Banana Food')].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==120) & (kg_allwaves['cropID']=='Beans')].index)
kg_allwaves = kg_allwaves.drop(kg_allwaves[(kg_allwaves['unit']==120) & (kg_allwaves['cropID']=='Maize')].index)



# For the case of consumption we will use the direct conversion rate (equally across items) for these units.

# merge with excel on units (manually introduced seeing the codebook from UNPS)
# also for missing ones (like a piece of fish, spoon), I use the measures from Malawi.
malawi_conversions = pd.read_stata("Conversion_kg_IHS4.dta", convert_categoricals=False)
med_units = malawi_conversions.groupby(by='unit').median()
# 20 teaspoon 0.006
# 59 tablespoon 0.015
# no info on fish kg... I can only make them up.


direct_units = pd.read_excel('cons_units_edit.xls')
direct_units.columns = ['unit', 'kgconverter_direct', 'unit_name']

kg_allwaves = kg_allwaves.merge(direct_units, on='unit', how='outer')

for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves[var].fillna(kg_allwaves['kgconverter_direct'], inplace=True)

kg_allwaves['kgconverter_med'] = kg_allwaves[['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']].median(axis=1, skipna=True)

kg_allwaves['kgconverter_med'].fillna(kg_allwaves['kgconverter_direct'],inplace=True)

kg_allwaves.loc[kg_allwaves['kgconverter_med']>160,'kgconverter_med'] = kg_allwaves.loc[kg_allwaves['kgconverter_med']>160,'kgconverter_direct']

### now let's manually detect wrong conversions and change them
for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves.loc[kg_allwaves[var]==0, var] = kg_allwaves.loc[kg_allwaves[var]==0, 'kgconverter_med']

for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves.loc[kg_allwaves[var]>160, var] = kg_allwaves.loc[kg_allwaves[var]==0, 'kgconverter_med']

for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves.loc[kg_allwaves[var]>2*kg_allwaves['kgconverter_med'], var] = kg_allwaves.loc[kg_allwaves[var]>2*kg_allwaves['kgconverter_med'], 'kgconverter_med']

for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves.loc[kg_allwaves[var]<kg_allwaves['kgconverter_med']/2, var] = kg_allwaves.loc[kg_allwaves[var]<kg_allwaves['kgconverter_med']/2, 'kgconverter_med']

# bundle-big
kg_allwaves.loc[kg_allwaves['unit']==96,['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 6

# container small
kg_allwaves.loc[kg_allwaves['unit']==130,['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 1


kg_allwaves.loc[(kg_allwaves['unit']==69) & (kg_allwaves['cropID']=='Sweet Potatoes'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 2
kg_allwaves.loc[(kg_allwaves['unit']==70) & (kg_allwaves['cropID']=='Sweet Potatoes'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 2
kg_allwaves.loc[(kg_allwaves['unit']==87) & (kg_allwaves['cropID']=='Tomatoes'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 1
kg_allwaves.loc[(kg_allwaves['unit']==89) & (kg_allwaves['cropID']=='Coffee All'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 2
kg_allwaves.loc[(kg_allwaves['unit']==90) & (kg_allwaves['cropID']=='Groundnuts'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 5
kg_allwaves.loc[(kg_allwaves['unit']==90) & (kg_allwaves['cropID']=='Tomatoes'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 5
kg_allwaves.loc[(kg_allwaves['unit']==90) & (kg_allwaves['cropID']=='Tobacco'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 5
kg_allwaves.loc[(kg_allwaves['unit']==93) & (kg_allwaves['cropID']=='Sorghum'),['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15','kgconverter_med']] = 50


for var in ['kgconverter_09','kgconverter_10','kgconverter_11','kgconverter_13','kgconverter_15']:
    kg_allwaves[var].fillna(kg_allwaves['kgconverter_med'],inplace=True)

# Use the list of items to manually matvh them with the list of consumption items.
tete =(kg_allwaves['cropID'].value_counts()).to_frame()
tete.reset_index(inplace=True)

#Without consumption units
kg_allwaves.to_csv('ag_conversionkg_allwaves.csv', index=False)


#%% With consumption units
c_codes = pd.read_csv('C:/Users/rodri/Dropbox/JMP/data/auxiliary data/c_items_codes.csv')
c_codes['cropID'].fillna(c_codes['item'], inplace=True)

kg_allwaves = kg_allwaves.merge(c_codes, how='outer', on= 'cropID')

# To fill non-agricultural consumption items use direct kg conversion. For the units that  
# do not have an obvious conversion, use median values.

median_vals = kg_allwaves[['unit','kgconverter_med']].groupby(by='unit').median()
median_vals.columns = ['kgconverter_unit_med']
median_vals.reset_index(inplace=True)

direct_units = direct_units.merge(median_vals, on='unit', how='left')

direct_units['kgconverter_direct'] = direct_units['kgconverter_direct'].fillna(direct_units['kgconverter_unit_med'])

# bundle (had extreme values)
direct_units.loc[direct_units['unit']==96, 'kgconverter_direct'] = 6
direct_units.loc[direct_units['unit']==97, 'kgconverter_direct'] = 3
direct_units.loc[direct_units['unit']==98, 'kgconverter_direct'] = 1

# Akendo: I don't have values and I don't know what it is. Yet better assign "reasonable" conversions than nan
direct_units.loc[direct_units['unit']==123, 'kgconverter_direct'] = 6
direct_units.loc[direct_units['unit']==124, 'kgconverter_direct'] = 3
direct_units.loc[direct_units['unit']==125, 'kgconverter_direct'] = 1

# Container: I don't have values and difficult to guess. Same as bundle and akendo.
direct_units.loc[direct_units['unit']==127, 'kgconverter_direct'] = 6
direct_units.loc[direct_units['unit']==128, 'kgconverter_direct'] = 3
direct_units.loc[direct_units['unit']==129, 'kgconverter_direct'] = 1

direct_units.to_csv('c_directkg_units.csv', index=False)






kg_allwaves.to_csv('conversionkg_allwaves.csv', index=False)