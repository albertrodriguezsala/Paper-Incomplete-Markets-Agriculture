# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:14:26 2019

@author: rodri
"""
# =============================================================================
# EMPIRICAL ANALYISIS HIGH CROPS AND LOW CROPS: 
# EMPIRICAL FINDINGS (2) AND (3). SUMMARY STATS, ESTIMATION, AND DATA MOMENTS CALIBRATION)
# =============================================================================

'''
Takes the panel plot-crop level data and produces
TABLES:
    - TABLE 2 PERCENTAGE GROWING CROPS
    - TABLE 11 AGRIC OUPTUT TO MARKET, OWN CONSUMPTION, STORED, GIFTS
    - TABLE 12 INTERMEDIATE INPUT SUMMARIES
    - TABLE 10: REGRESSION SHARE HIGH VS LOW CROPS ON CIW
FIGURE 2: SHARES CROPS ACROSS WEALTH DISTRIBUTION
    
ESTIMATES:
    - ESTIMATES THE AR(1) PROCESS ON NON-AGRICULTURAL INCOME
    - ESTIMATES THE VOLATILITY OF THE HIGH CROPS AND LOW CROPS.
    - ESTIMATES THE VOLATILITY ON CONSUMPTION AND INCOME.

DATA MOMENTS:
    OUTPUT HIGH AND LOW CROPS, RISK HIGH CROPS, RISK LOW CROPS.
    CONSUMPTION VOLATILITY, INCOME VOLATILITY.
'''
# produces figure 2, table 2. summary statistics low vs high.

# calibrate sigmas in agricultural shocks in the high and low crops.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col

pd.options.display.float_format = '{:,.2f}'.format
from matplotlib.ticker import FormatStrFormatter
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf
save=False
dollars = 2586.89

from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
import sys
sys.path.append(str(dirct))
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

folder =  dirct+'/data/panel/'
folder_fig= dirct+'/Results/figures/Uganda stats/'




percentiles = [0.5]

### All monetary variables controlled for inflation and in 2013 US$
# Farm capital not in dollars (but k yes)


import warnings
warnings.filterwarnings("ignore")


### Graphic aesthetics
import matplotlib as mpl
fm = mpl.font_manager

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.6, 6.4)
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
colormap = plt.cm.Dark2.colors


# Import data ==============================================

data = pd.read_csv(folder+'panel_plotcrop_data.csv')

data['cropID'] = data['cropID'].replace('Banana Food','Plantain Bananas')


#data.replace(0, np.nan, inplace=True)
data.set_index(['HHID', 'plotID', 'cropID', 'wave', 'season'], inplace=True)
data.reset_index(inplace=True)


## Generate model variables:

# NO LONG-TERM ONES. Select crops based on households-seasonal variation: min obs per crop = 50
### COCOA takes at least 3-5 years: https://www.cocoalife.org/in-the-cocoa-origins
### MANGO trees at least 3 years to produce food.
### pineapples is 1-1.5 years. let is consider it as short.
## AVOCADO NOT THAT MUCH INFO BUT TAKE IT AWAY
## PAW PAW 4-8 years
## SIMSIM = SESAME. Grows in a season.
## COTTON 6 months
## VANILLA 3 years
## TEA: 3 to 15 years
## passion fruit: 2-3 years


# only time volatility -------------------
crops_high = ['Avocado', 'Plantain Bananas', 'Banana Sweet', 'Cabbage', 'Cassava',  'Eggplants', 'Groundnuts', 'Jackfruit',
              'Mango', 'Oranges', 'Paw Paw', 'Pumpkins', 'Tomatoes', 'Yam', 'Rice', 'Sweet Potatoes', 'Tobacco']

crops_low = [ 'Banana Beer',  'Beans', 'Coffee All', 'Cotton', 'Field Peas', 'Finger Millet', 'Irish Potatotes', 'Maize',
         'Onions','Pigeon Peas', 'Pineapples', 'Simsim', 'Sorghum', 'Soya Beans', 'Sunflower', 'Sugarcane']


## No long term crops
crops_high = ['Banana Beer','Plantain Bananas', 'Banana Sweet',  'Cabbage', 
              'Eggplants', 'Groundnuts', ' Irish Potatoes', 'Onions', 'Pineapples', 'Pumpkins', 'Rice', 'Sugarcane', 'Tomatoes', 'Yam']

crops_low = ['Beans', 'Cassava', 'Cotton', 'Dodo', 'Cow Peas', 'Field Peas', 'Finger Millet', 
  'Maize', 'Pigeon Peas', 'Simsim','Sorghum','Soya Beans','Sunflower', 'Sweet Potatoes']

print('List of high crops:')
print(crops_high)
print('List of low crops:')
print(crops_low)

#### FOR PLOTS ON SPECIFIC CROPS
crops_var_high = ['Plantain Bananas', 'Banana Beer', 'Rice', 'Tomatoes',  'Groundnuts',  'Banana Sweet']
crops_var_low = ['Beans', 'Soya Beans', 'Cassava', 'Simsim', 'Sorghum' , 'Finger Millet', 'Maize']
    
for var in crops_high:
    data['y_'+var] = data.loc[data['cropID'] == var, 'y']
    
for var in crops_low:
    data['y_'+var] = data.loc[data['cropID'] == var, 'y']    
 
variables = [ 'lny_over_A', 'lnm', 'lnl',  'y_over_A', 'm', 'l',  'y', 'A', 'lny', 'lnA', 'area_planted']

for var in variables:
    data[var+'_h'] = data.loc[data['cropID'].isin(crops_high), var]
    data[var+'_l'] = data.loc[data['cropID'].isin(crops_low), var]
     

## PANEL
data['t'] = np.nan
data.loc[(data['wave']==2009) & (data['season']==2),'t'] = 1
data.loc[(data['wave']==2009) & (data['season']==1),'t'] = 2
data.loc[(data['wave']==2010) & (data['season']==2),'t'] = 3
data.loc[(data['wave']==2010) & (data['season']==1),'t'] = 4
data.loc[(data['wave']==2011) & (data['season']==2),'t'] = 5
data.loc[(data['wave']==2011) & (data['season']==1),'t'] = 6
data.loc[(data['wave']==2013) & (data['season']==2),'t'] = 7
data.loc[(data['wave']==2013) & (data['season']==1),'t'] = 8
data.loc[(data['wave']==2015) & (data['season']==2),'t'] = 9
data.loc[(data['wave']==2015) & (data['season']==1),'t'] = 10

data.rename(columns={"HHID":"hh"}, inplace=True)
 


#%% Intermediate inputs and production high crops and low crops statistics


count_crops = pd.value_counts(data['cropID'])
data_hhs = data.groupby(by=['hh','wave']).sum().replace(0,np.nan)
data_hhs['A'] = data_hhs['A']/2




print('    ')
print('===============================')
print(' TABLE IN APPENDIX: AGRICULTURAL PRODUCTION TO MARKET, OWN CONSUMPTION, STORED, GIFTED')
## summary 1: share of production to selling, own-consumption. etc
data_hhs['sell_share'] = data_hhs['sell_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['cons_share'] = data_hhs['cons_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['store_share'] = data_hhs['stored_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['gift_share'] = data_hhs['gift_kg'].fillna(0)/data_hhs['total_kg']
data_hhs['interm_share'] = (data_hhs['animal_kg'].fillna(0)+data_hhs['food_prod_kg'].fillna(0)+data_hhs['seeds_kg'].fillna(0))/data_hhs['total_kg']

data_hhs['sell_yes'] = 1*(data_hhs['sell_kg']>10)
data_hhs['cons_yes'] = 1*(data_hhs['cons_kg']>10)
data_hhs['store_yes'] = 1*(data_hhs['stored_kg']>10)
data_hhs['gift_yes'] = 1*(data_hhs['gift_kg']>10)



data_hhs[['y','A','total_kg','m']].describe(percentiles=percentiles)


data_hhs.reset_index(inplace=True)


sum_waves_yshares = 100*(data_hhs[['wave','sell_share','cons_share','store_share','gift_share','sell_yes','cons_yes','store_yes','gift_yes']].groupby(by='wave').mean())
print(sum_waves_yshares.to_latex())



data_hh = data.groupby(by=['hh','wave']).sum()
data_hh.reset_index(inplace=True)

# Summary intermediate inputs
data_hhs_m = data_hh.loc[(data_hh['A']>0.1) & (data_hh['y']>50)]
sum_m_shares = data_hhs_m[['wave','m', 'm_h','m_l','chem_fert', 'org_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].groupby(by='wave').agg(lambda x: (x > 0).mean())*100
sum_m_value = (data_hhs_m[['wave','m', 'm_h', "m_l",'chem_fert', 'org_fert', 'pesticides','seed_cost','trans_cost','labor_payment']].replace([0,0.0],np.nan)).groupby(by='wave').mean()

a = sum_m_shares.mean(axis=0)
b = sum_m_value.mean(axis=0)

sum_m = pd.concat([a, b], axis=1).T

print('    ')
print('===============================')
print('TABLE IN APPENDIX: INTERMEDIATE INPUTS SUMMARY')
print(sum_m.iloc[1,0:3].to_latex())
print(sum_m.iloc[:,3:-1].to_latex())



## Do households grow 2 type of crops? table 2

# remove "outliers" 
data_hh['high_crop'] = 1*(data_hh['y_h']>1)
data_hh['high_crop_A'] = 1*(data_hh['area_planted_h']>0.01)
data_hh['low_crop'] = 1*(data_hh['y_l']>1)
data_hh['low_crop_A'] = 1*(data_hh['area_planted_l']>0.01) 
data_hh['both_crops'] = 1*((data_hh['high_crop']+data_hh['low_crop']==2))
data_hh['both_crops_A'] = 1*((data_hh['high_crop_A']+data_hh['low_crop_A']==2))
 

# compute proportion of crops grown across hh in each wave table 2
crop_choice_tab = []
for wave in [2009, 2010.0, 2011, 2013, 2015]:
    #for season in [1.0,2.0]:
    crop_choice_tab.append((data_hh.loc[(data_hh['wave']==wave), ['high_crop','low_crop','both_crops']].describe()).iloc[1,:]*100)
 
crop_choice_data = pd.DataFrame(crop_choice_tab)
crop_choice_data.loc['mean_waves'] = crop_choice_data.mean()

print('    ')
print('===============================')
print(' TABLE 2: SHARE OF FARMERS CULTIVATING CROPS')
print(crop_choice_data.to_latex())


panel = data.groupby(by=['hh','wave']).sum()
panel.reset_index(inplace=True)
panel.replace(0, np.nan, inplace=True)
count_hh = panel.groupby(by='hh')[['wave']].count()
count_hh.columns = ['count_hh']
panel = panel.merge(count_hh, on='hh')
balpanel = panel.loc[panel['count_hh']==4]


# total land from low and high crops
panel['A'] = panel['A_h'].fillna(0) + panel['A_l'].fillna(0)
panel['A'].describe(percentiles=percentiles)
panel['l'] = panel['l_h'].fillna(0) + panel['l_l'].fillna(0)

panel = panel.loc[panel['A']>0]



##### shares output crops, average production and input usage high crops vs low crops.
panel_avg = panel.groupby(by=['hh', 'wave']).sum()
panel_avg['yh_yes_wave'] = 1*(panel_avg['y_h']>1)
panel_avg['yl_yes_wave'] = 1*(panel_avg['y_l']>1)

panel_avg[['y_h','y_l','A_h','A_l','m_h','m_l']].replace(0,np.nan)

panel_avg = panel_avg.groupby(by='hh').mean()

panel_avg['y'] = panel_avg[['y_h','y_l']].sum(axis=1)
panel_avg['yh_share'] = panel_avg['y_h'].replace(0,np.nan)/panel_avg['y'].replace(0,np.nan)

panel_avg['yh_yes'] = 1*(panel_avg['y_h']>1)
panel_avg['yl_yes'] = 1*(panel_avg['y_l']>1)
panel_avg['yhl_yes'] = 1*((panel_avg['yh_yes']==1) & (panel_avg['yl_yes'] ==1))

shares_allwaves = (np.mean(panel_avg[['yh_yes','yl_yes','yhl_yes']])*100).T




print('LAST ROW TABLE 2: GREW HIGH AND LOW CROPS EACH/IN WAVES')
print(shares_allwaves)



sum_waves = panel[['wave','y_h','y_l', 'A_h','A_l', 'm_h','m_l']].groupby(by='wave').mean()

sum_waves_mean = sum_waves.mean(axis=0)
sum_waves= sum_waves.append(sum_waves_mean, ignore_index=True)
print('    ')
print('===============================')
print('TABLE IN ONLINE APPENDIX. summary low vs high crops. ')
print(sum_waves.to_latex())


print('    ')
print('===============================')
print(' DATA MOMENTS:  average production and input usage high crops and low crops.')
print('1. avg(y_h)  2.avg(y_l)     3.avg(m_h)    4.avg(m_l):')
print(sum_waves.iloc[-1,[0,1,4,5]].T)
print('y_h/y_l',)
 




#%% UPLOAD HOUSEHOLD PANEL: 

## upload now household information (for production along wealth distribution and for regressions)
panel_UGA =pd.read_csv(folder+'panel_rural_UGA.csv')

panel_UGA.replace(['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016'],[2009, 2010.0, 2011, 2013, 2015],inplace=True)


#%% Non-agricultural income process

print('    ')
print('===============================')
print('ESTIMATING THE AR(1) NON-AGRICULTURAL INCOME PROCESS')


# generante non agric earnings
panel_UGA['y_noagric'] = panel_UGA[['profit_lvstk', 'bs_profit', 'wage_total' ]].sum(axis=1)
panel_UGA['y_noagric'].replace(0,np.nan,inplace=True)
panel_UGA['ln_yna'] = np.log(panel_UGA['y_noagric'])
panel_UGA['y_noagric_lag'] = panel_UGA.groupby(by='hh')['y_noagric'].shift(1) 
panel_UGA['ln_yna_lag'] = panel_UGA.groupby(by='hh')['ln_yna'].shift(1) 

del panel_UGA['A'], panel_UGA['l']

panel =  panel.merge(panel_UGA, on=['hh','wave'], how='left')
balpanel =  balpanel.merge(panel_UGA, on=['hh','wave'], how='left')
panel.y_noagric.describe()

panel_yna = panel.dropna(subset=['ln_yna','ln_yna_lag'])
panel_yna.set_index(['hh','wave'], inplace=True)
ols_yna = PanelOLS.from_formula('ln_yna ~ 1+ ln_yna_lag  ', data=panel_yna).fit(cov_type='robust')
print(ols_yna)

# sipmle OLS
import statsmodels.api as sm
Y = panel_yna['ln_yna']
X = panel_yna['ln_yna_lag']
# Add a constant term to the independent variable (intercept)
X = sm.add_constant(X)
# Fit the AR(1) model
model = sm.OLS(Y, X).fit()
# Get the model summary

# gived the same estimates as previous one.


# autocorr: 0.398

residual_yna =ols_yna.idiosyncratic
sd_yna = np.sqrt(np.var(residual_yna))

print('    ')
print('===============================')
print('ESTIMATED PARAMETERS')
print('rho and sigma AR(1) in log non-agricultural income')
print('rho:',ols_yna.params[1])
print('sigma:',sd_yna)
# note that later on I add these variables as well





#%% Risk moments: 
# targeted: High crops risk, low crops risk, and risk-correlation. 
# untargeted: income and consumption risk.
# using volatility on residual measures



#generatre log variables.
variables = ['inctotal','ctotal', 'A','l', 'm_l', 'l_l', 'A_l', 'y_l',  'm_h', 'l_h', 'A_h', 'y_h']
for var in variables:
    panel['ln'+var] =  np.log(panel[var].dropna()+np.abs(np.min(panel[var]))).replace(-np.inf, np.nan)

    

print('===========================================')
print('Computing risk moments crops:')
print('1. high crops risk, 2. low crops risk, 3. corr(low crops, high crops)')

print('     ')
print('1st step. Regression output crops on vars not in model')
## high crops
panel_h = panel.dropna(subset=['y_h','A_h'])
panel_h.set_index(['hh','wave'], inplace=True)
# 1st step: remove factors not in the model andn ot about risk
ols_outph = PanelOLS.from_formula('lny_h ~ lnA_h  +lnl_h  +sex +classeduc  +age +age_sq +familysize +EntityEffects   +TimeEffects ', data=panel_h).fit(cov_type='robust')
print(ols_outph)

residual_h =ols_outph.idiosyncratic
yh_hat = ols_outph.fitted_values
yh_hat = yh_hat.merge(panel_h['lny_h'], on=['hh','wave'])
yh_hat['u_h'] = np.abs(yh_hat['lny_h'] - yh_hat['fitted_values'])
sd_theta = np.sqrt(np.var(residual_h))
sd_theta2 = np.sqrt(np.var(yh_hat['u_h']))  

panel_l = panel.dropna(subset=['y_l','A_l'])
panel_l.set_index(['hh','wave'], inplace=True)
ols_outpl = PanelOLS.from_formula('lny_l ~ lnA_l +lnl_l  +sex +classeduc +age   +familysize   +EntityEffects  +TimeEffects', data=panel_l).fit(cov_type='robust')
print(ols_outpl)


residual_l =ols_outpl.idiosyncratic
sd_eps = np.sqrt(np.var(ols_outpl.resids))
yl_hat = ols_outpl.fitted_values
yl_hat = yl_hat.merge(panel_l['lny_l'], on=['hh','wave'])
yl_hat['u_l'] = np.abs(yl_hat['lny_l'] - yl_hat['fitted_values'])

print('===========================================')





print('2nd step. Compute volatility residuals')
data_residuals = yh_hat.merge(yl_hat, how='outer', on=['hh','wave'])
timevar = data_residuals.groupby(by='hh')[['u_h', 'u_l']].cov()
timevar.reset_index(inplace=True)
blu = timevar.groupby(by='level_1').mean()
sd_yh = np.sqrt(blu.iloc[0,1])
sd_yl = np.sqrt(blu.iloc[1,2])


cov_yhyl = blu.iloc[1,1]/(sd_yh*sd_yl)
list_moments = [round(blu.iloc[0,1],4), round(blu.iloc[1,2],4), round(cov_yhyl,4)]  

print('     ')
print('===========================================')
print('TO TARGET MOMENTS ON RISK')
print('1. high crops risk, 2. low crops risk, 3. corr(low crops, high crops:')
print(list_moments)
print('     ')



print('===========================================')
print('Computing income risk and consumption risk:')

print('     ')
print('1st step. Regression income/consumption on vars not in model')
panel_y = panel.dropna(subset=['lninctotal'])
panel_y.set_index(['hh','wave'], inplace=True)

# setp 1
ols_y = PanelOLS.from_formula('lninctotal ~ lnA  +sex +classeduc  +age +age_sq +familysize +EntityEffects   +TimeEffects ', data=panel_y).fit(cov_type='robust')
print(ols_y)
residual_h =ols_y.idiosyncratic
y_hat = ols_y.fitted_values
y_hat = y_hat.merge(panel_y['lninctotal'], on=['hh','wave'])
y_hat['u_y'] = np.abs(y_hat['lninctotal'] - y_hat['fitted_values'])
sd_theta = np.sqrt(np.var(residual_h))
sd_theta2 = np.sqrt(np.var(y_hat['u_y']))  


panel_c = panel.dropna(subset=['lnctotal'])
panel_c.set_index(['hh','wave'], inplace=True)
ols_c = PanelOLS.from_formula('lnctotal ~ lnA  +sex +classeduc +age   +familysize   +EntityEffects  +TimeEffects', data=panel_l).fit(cov_type='robust')
print(ols_c)


residual_c =ols_c.idiosyncratic
sd_eps = np.sqrt(np.var(ols_c.resids))
c_hat = ols_c.fitted_values
c_hat = c_hat.merge(panel_c['lnctotal'], on=['hh','wave'])
c_hat['u_c'] = np.abs(c_hat['lnctotal'] - c_hat['fitted_values'])

data_residuals = y_hat.merge(c_hat, how='outer', on=['hh','wave'])

# step 2
timevar = data_residuals.groupby(by='hh')[['u_y', 'u_c']].cov()
timevar.reset_index(inplace=True)


blu = timevar.groupby(by='level_1').mean()


print('     ')
print('===========================================')
print('NON-TARGET RISK MOMENTS')
print('Income risk: ',round(blu.iloc[1,1],4))
print('Consumption risk: ',round(blu.iloc[0,2],4))
print('===========================================')
print('     ')
  

panel_w = panel_UGA[['hh','wtotal','liquid_w']].groupby(by='hh').mean()
panel_w.reset_index(inplace=True)

timevar2 = timevar.groupby(by='hh').mean()

timevar2.mean()

vol_in_w = timevar.merge(panel_w, on='hh')

#vol_in_w.to_csv('ci_volatility.csv')




#%% Crops production along the wealth distribution
del panel['inctotal'], panel['ctotal'], panel['wtotal']

save= True
panel_ciw = pd.read_csv(folder+'panel_rural_UGA.csv')

data_cwi = panel_ciw[['hh','wave','inctotal', 'ctotal', 'wtotal']]
data_cwi.replace(['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016'], [2009.0, 2010.0, 2011.0, 2013.0, 2015.0], inplace=True)
data_cwi_avg = data_cwi.groupby(by='hh').mean()
data_cwi_avg.reset_index(inplace=True)

#data_cwi_avg = data_cwi.loc[data_cwi['wave']==2011]
 
# =============================================================================
# ####  empirical evidence rich households invest more in high productive:
# =============================================================================
agric_hh = panel.groupby(by=['hh', 'wave']).sum()

agric_hh['y'] = (agric_hh['y_h'].fillna(0) + agric_hh['y_l'].fillna(0)).replace(0,np.nan)
agric_hh['A'] = (agric_hh['A_h'].fillna(0) + agric_hh['A_l'].fillna(0)).replace(0,np.nan)
 
agric_hh['yh_over_y'] = (agric_hh['y_h'].divide(agric_hh['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
agric_hh['yl_over_y'] = (agric_hh['y_l'].divide(agric_hh['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
 
data_cwi = agric_hh.merge(data_cwi_avg, on=['hh'], how='left')
 
## generate variables in logs:
var_list = ['y','A','y_h', 'y_l', 'inctotal', 'ctotal', 'wtotal']
for var in var_list:
    data_cwi['ln'+var] = np.log((data_cwi[var]+np.min(data_cwi[var])).replace(0,np.nan))
 

data_cwi.to_csv(folder+'panelrural_highlowcrops.csv')
    
### estimating elasticities CIW distributions and share of production coming from high crops wrt low crops.
## first Regression table in the appendix.
model_c= smf.ols(formula = "lny_h - lny ~ 1 +lnctotal", data=data_cwi)
ols_c = model_c.fit()

 
model_A= smf.ols(formula = "lny_h - lny ~ 1 +lnA", data=data_cwi)
ols_A = model_A.fit()


model_w= smf.ols(formula = "lny_h - lny ~ 1 +lnwtotal", data=data_cwi)
ols_w = model_w.fit()

 
model_i= smf.ols(formula = "lny_h - lny ~ 1 +lninctotal", data=data_cwi)
ols_i = model_i.fit()

 
results = summary_col([ols_c, ols_i, ols_w, ols_A],stars=False, info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)}, regressor_order=['lnctotal','lninctotal',  'lnwtotal', 'lnA'])

print('     ')
print('===========================================')
print(' TABLE IN APPENDIX: ELASTICITIES CIW IN HIGH VS LOW CROPS')
print(results.as_latex())


 
 
print('     ')
print('===========================================')
print('FIGURE 2: CROP PRODUCTION ALONG THE WEALTH DISTRIBUTION')
 
percentiles = np.linspace(0,100,21)

n_p = len(percentiles)
data_y = data_cwi.groupby(pd.cut(data_cwi.wtotal, np.percentile(data_cwi['wtotal'].dropna(), percentiles), include_lowest=True)).mean()

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
 

 

data_y['y'] = (data_y['y_h'].fillna(0) + data_y['y_l'].fillna(0)).replace(0,np.nan)
data_y['yh_over_y'] = (data_y['y_h'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
data_y['yl_over_y'] = (data_y['y_l'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)




print('     ')
print('===========================================')
print('FIGURE 2A) HIGH VS LOW CROPS')
fig, ax = plt.subplots()
ax.plot(percentiles[1:n_p], data_y['yh_over_y'], label = 'High crops', color=colormap[0])
ax.plot(percentiles[1:n_p], data_y['yl_over_y'], label = 'Low crops', color=colormap[1])
ax.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(percentiles[1:n_p])
ax.set_xlabel('Wealth Distribution')
#ax.set_title('Low vs High Crops Output Shares along Wealth')
ax.set_ylabel('Crop Output over Total (Share)')
if save==True:
    fig.savefig(folder_fig+'crops_along_wealth.png')
plt.plot()

#### Check per each crop ------------------------------------

n_p = len(percentiles)
data_y = data_cwi.groupby(pd.cut(data_cwi.wtotal, np.percentile(data_cwi['wtotal'].dropna(), percentiles), include_lowest=True)).mean()
 

for var in crops_low:
    data_y[var+'_over_y'] = (data_y['y_'+var].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
  

for var in crops_high:
    data_y[var+'_over_y'] = (data_y['y_'+var].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
 
 
print('     ')
print('===========================================')
print('FIGURE 2b) HIGH CROPS')
colormap = plt.cm.Dark2.colors
fig, ax = plt.subplots()
for crop in crops_var_high:
    ax.plot(percentiles[1:n_p], data_y[crop+'_over_y'], label = crop)
ax.legend(loc='upper left')
plt.xticks(percentiles[1:n_p])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel('Crop Output over Total (Share)')
#ax.set_title('High Crops Production along Wealth Distribution')
ax.set_xlabel('Wealth Distribution')
if save==True:
    fig.savefig(folder_fig+'cropshigh_along_wealth.png')
plt.plot()
 
 
print('     ')
print('===========================================')
print('FIGURE 2b) LOW CROPS')
colormap = plt.cm.Dark2.colors
fig, ax = plt.subplots()
for crop in crops_var_low:
    ax.plot(percentiles[1:n_p], data_y[crop+'_over_y'], label = crop)
ax.legend(loc='upper right')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(percentiles[1:n_p])
ax.set_ylabel('Crop Output over Total (Share)')
ax.set_xlabel('Wealth Distribution')
if save==True:
    fig.savefig(folder_fig+'cropslow_along_wealth.png')
plt.plot()


#%% ROBUSTNESS: INCOME DISTRIBUTION. FIGURE XX IN APPENDIX

percentiles = np.linspace(0,100,21)

n_p = len(percentiles)
data_y = data_cwi.groupby(pd.cut(data_cwi.inctotal, np.percentile(data_cwi['inctotal'].dropna(), percentiles), include_lowest=True)).mean()
 
 
print('     ')
print('===========================================')
print('FIGURE IN APPENDIX: CROP PRODUCTION ALONG THE INCOME DISTRIBUTION') 

data_y['y'] = (data_y['y_h'].fillna(0) + data_y['y_l'].fillna(0)).replace(0,np.nan)
data_y['yh_over_y'] = (data_y['y_h'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
data_y['yl_over_y'] = (data_y['y_l'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)


 
 
#data_y = data_cwi.groupby(by='y').mean()
#data_y.reset_index(inplace=True)
fig, ax = plt.subplots()
ax.plot(percentiles[1:n_p], data_y['yh_over_y'], label = 'High crops', color=colormap[0])
ax.plot(percentiles[1:n_p], data_y['yl_over_y'], label = 'Low crops', color=colormap[1])
ax.legend()
plt.xticks(percentiles[1:n_p])
ax.set_xlabel('Income Distribution')
#ax.set_title('Low vs high crops production shares along wealth (UNPS 09-11)')
ax.set_ylabel('Crop Output over Total (Share)')
if save==True:
    fig.savefig(folder_fig+'crop_along_income.png')
plt.plot()

#percentiles = [0.05, 0.2, 0.4, 0.6, 0.8, 1]
n_p = len(percentiles)
data_y = data_cwi.groupby(pd.cut(data_cwi.inctotal, np.percentile(data_cwi['inctotal'].dropna(), percentiles), include_lowest=True)).mean()
 

for var in crops_low:
    data_y[var+'_over_y'] = (data_y['y_'+var].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
  

for var in crops_high:
    data_y[var+'_over_y'] = (data_y['y_'+var].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
 
   

 
colormap = plt.cm.Dark2.colors
fig, ax = plt.subplots()
for crop in crops_var_high:
    ax.plot(percentiles[1:n_p], data_y[crop+'_over_y'], label = crop)
ax.legend(loc='upper left')
plt.xticks(percentiles[1:n_p])

ax.set_ylabel('Crop Output over Total (Share)')
#ax.set_title('High crops production along wealth distribution (UNPS 09-16)')
ax.set_xlabel('Income Distribution')
if save==True:
    fig.savefig(folder_fig+'cropshigh_along_income.png')
plt.plot()
 
 
 
colormap = plt.cm.Dark2.colors
fig, ax = plt.subplots()
for crop in crops_var_low:
    ax.plot(percentiles[1:n_p], data_y[crop+'_over_y'], label = crop)
ax.legend(loc='upper right')
plt.xticks(percentiles[1:n_p])
ax.set_ylabel('Crop Output over Total (Share)')
#ax.set_title('Low crops production along wealth distribution (UNPS 09-16)')
ax.set_xlabel('Income Distribution')
if save==True:
    fig.savefig(folder_fig+'cropslow_along_income.png')
plt.plot()






