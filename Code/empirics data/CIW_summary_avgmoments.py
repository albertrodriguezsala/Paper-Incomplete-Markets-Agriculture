# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:50:07 2023

@author: rodri
"""

##### CIW SUMMARY TABLES

import pandas as pd
import numpy as np
import os
pd.options.display.float_format = '{:,.2f}'.format
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers, gini, data_stats
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

my_dirct =  dirct+'/data/panel/'

## IMPORT PANELS: NATIONWIDE AND ONLY RURAL
panel =pd.read_csv(my_dirct+"panel_UGA.csv")
panel_rural = pd.read_csv(my_dirct+"panel_rural_UGA.csv")


import warnings
warnings.filterwarnings("ignore")
#%% =============================================================================
# Agricultural output and liquid wealth: average and Gini
# =============================================================================
pd.options.display.float_format = '{:,.3f}'.format
print('    ')
print('=========================================')
print(' Empricial moments agriculture distrib. and liquid wealth targeted calibration ')

summary = []
gini_list = []
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']
var_list = ['y_agric','liquid_w']

for wave in waves:
    summary.append(np.mean(panel_rural.loc[panel_rural['wave']==wave, ['y_agric','liquid_w']], axis=0))
    gini_wave= []
    for var in var_list:
        gini_wave.append(gini((panel_rural.loc[panel_rural['wave']==wave,var].replace([np.inf, -np.inf],np.nan)).dropna()))
    gini_list.append(gini_wave)
    

avgs_df = pd.DataFrame(summary)
gini_df = pd.DataFrame(gini_list)

rural_mean = avgs_df.mean(axis=0)
rural_gini = gini_df.mean(axis=0)

print('average liquid wealth (trgt)')
print(rural_mean)
print('Gini agric output (trgt) and liquid wealth')
print(rural_gini)

pd.options.display.float_format = '{:,.2f}'.format

#%% ===========================================================================
# CIW: AVERAGE AND GINI ACROSS WAVES
# =============================================================================

print('=========================================')
print(' TABLE IN APPENDIX:  CIW SUMMARY ')

### National aggregate:
print('hh averages')
summary = []
gini_list = []
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']
var_list = ['ctotal','inctotal', 'wtotal']
mean_lw = []
for wave in waves:
    summary.append(np.mean(panel.loc[panel['wave']==wave, ['ctotal','inctotal','wtotal']], axis=0))
    gini_wave= []
    for var in var_list:
        gini_wave.append(gini((panel.loc[panel['wave']==wave,var].replace([np.inf, -np.inf],np.nan)).dropna()))
    gini_list.append(gini_wave)
    

sum_ciw = pd.DataFrame(summary)
sum_gini = pd.DataFrame(gini_list)

mean_cwi = pd.DataFrame(sum_ciw.mean(axis=0)).T
mean_gini =pd.DataFrame(sum_gini.mean(axis=0)).T
sum_ciw = pd.concat([sum_ciw,mean_cwi], ignore_index=True)
sum_gini = pd.concat([sum_gini,mean_gini], axis=0)
sum_ciw.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']
sum_gini.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']

# For Rural households:
summary = []
gini_list = []
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']
var_list = ['ctotal','inctotal', 'wtotal']

for wave in waves:
    summary.append(np.mean(panel_rural.loc[panel_rural['wave']==wave, ['ctotal','inctotal','wtotal']], axis=0))
    gini_wave= []
    for var in var_list:
        gini_wave.append(gini((panel_rural.loc[panel_rural['wave']==wave,var].replace([np.inf, -np.inf],np.nan)).dropna()))
    gini_list.append(gini_wave)
    

sum_ciw_rural = pd.DataFrame(summary)
sum_gini_rural = pd.DataFrame(gini_list)

rural_mean_cwi = pd.DataFrame(sum_ciw_rural.mean(axis=0)).T
rural_mean_gini = pd.DataFrame(sum_gini_rural.mean(axis=0)).T

sum_ciw_rural =  pd.concat([sum_ciw_rural,rural_mean_cwi], axis=0)
sum_gini = pd.concat([sum_gini,mean_gini], axis=0)
sum_gini_rural = pd.concat([sum_gini_rural,rural_mean_gini], axis=0)
sum_ciw_rural.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']
sum_gini_rural.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']


##urban and rural
waves_mean_cwi = pd.concat([sum_ciw_rural, sum_ciw], axis=1)


print(waves_mean_cwi.to_latex())
waves_sum_gini = pd.concat([sum_gini_rural, sum_gini], axis=1)

waves_sum_gini = '(' + round(waves_sum_gini,2).astype(str) +')'

print('gini coefficients')
print(waves_sum_gini.to_latex())



#%%
### PRESENT SAME TABLE BUT PER CAPITA VARIABLES

print('=========================================')
print(' TABLE IN APPENDIX:  CIW SUMMARY PER CAPITA LEVEL ')

summary = []
gini_list = []
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']
var_list = ['ctotal_cap','inctotal_cap', 'wtotal_cap']

for wave in waves:
    summary.append(np.mean(panel.loc[panel['wave']==wave, ['ctotal_cap','inctotal_cap','wtotal_cap','familysize']], axis=0))
    gini_wave= []
    for var in var_list:
        gini_wave.append(gini((panel.loc[panel['wave']==wave,var].replace([np.inf, -np.inf],np.nan)).dropna()))
    gini_list.append(gini_wave)
    
sum_ciw = pd.DataFrame(summary)
sum_gini = pd.DataFrame(gini_list)

mean_cwi = sum_ciw.mean(axis=0)
mean_gini = sum_gini.mean(axis=0)

sum_ciw = sum_ciw.append(mean_cwi, ignore_index=True)
sum_gini = sum_gini.append(mean_gini, ignore_index=True)
sum_ciw.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']
sum_gini.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']

# For Rural households:
panel_rural = panel.loc[panel['urban']==0]

summary = []
gini_list = []
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']
var_list = ['ctotal_cap','inctotal_cap', 'wtotal_cap', 'familysize']

for wave in waves:
    summary.append(np.mean(panel_rural.loc[panel_rural['wave']==wave, ['ctotal_cap','inctotal_cap','wtotal_cap', 'familysize']], axis=0))
    gini_wave= []
    for var in var_list:
        gini_wave.append(gini((panel_rural.loc[panel_rural['wave']==wave,var].replace([np.inf, -np.inf],np.nan)).dropna()))
    gini_list.append(gini_wave)

sum_ciw_rural = pd.DataFrame(summary)
sum_gini_rural = pd.DataFrame(gini_list)

rural_mean_cwi = sum_ciw_rural.mean(axis=0)
rural_mean_gini = sum_gini_rural.mean(axis=0)

sum_ciw_rural = sum_ciw_rural.append(rural_mean_cwi, ignore_index=True)
sum_gini_rural = sum_gini_rural.append(rural_mean_gini, ignore_index=True)
sum_ciw_rural.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']
sum_gini_rural.index =  ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016', 'Average']


##urban and rural
waves_mean_cwi = pd.concat([sum_ciw_rural, sum_ciw], axis=1)

waves_sum_gini = pd.concat([sum_gini_rural, sum_gini], axis=1)

waves_sum_gini = '(' + round(waves_sum_gini,2).astype(str) +')'

print(waves_mean_cwi.to_latex())
print('ginis')
print(waves_sum_gini.to_latex())





