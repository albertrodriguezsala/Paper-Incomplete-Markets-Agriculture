# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:29:34 2018

@author: rodri
"""

# =============================================================================
# Crops summary, risk and productivity analysis: all 5 waves, data at hh-plot-crop level
# =============================================================================

text = '''
Takes the panel plot-crop level data and produces
(1)
Figure 1: Crop Yields vs. Risk
Table 1: Robustness Crop Yields vs. Risk
Table 11 appendix: Shares Selling Crops
Table 1 and 2 in the online appendix

(2) Classifies crops in the High Crops vs. Low Crops based on Figure 1. 
Low Crops: Below Median. High Crops: Above Median

'''
import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers, gini
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

folder =  dirct
folder_fig= folder+'/Results/figures/Uganda stats/'


pd.options.display.float_format = '{:,.2f}'.format

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.6, 6.4)
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

dollars = 2586.89


data = pd.read_csv(folder+'/data/panel/panel_plotcrop_data.csv')

### The average across plots was 1.16
count_cropregion = data.groupby(by=['cropID','wave', 'region'])[['y']].count()
count_cropregion.columns = ['count']
count_cropregion.reset_index(inplace=True)

data['cropID'] = data['cropID'].replace('Banana Food','Plantain Bananas')

#%% Risk vs Yields Crops: Variation only across time: No long-term crops. Figure 1 and column 2, rows 3-4) table 1

print('     ')


crops_long = ['Avocado','Cocoa', 'Mango', 'Coffee All', 'Paw Paw', 'Tobacco', 'Passion Fruit', 'Oranges', 'Vanilla', 'Jackfruit']

data_nolong = data.loc[~data['cropID'].isin(crops_long)]

count_crop = data.groupby(by=['cropID','wave'])[['y']].count()
count_crop.columns = ['count_hh']
count_crop.reset_index(inplace=True)

count_crop2 = data.groupby(by=['cropID','wave','season'])[['y']].count()
count_crop2 = count_crop2.groupby(by=['cropID'])[['y']].max()
count_crop2.columns = ['Obs']
count_crop2.reset_index(inplace=True)


agg_data = data_nolong[['cropID','wave','y_over_A','y','A','m','l']].groupby(by=['cropID', 'wave']).mean()
agg_data.reset_index(inplace=True)
agg_data = agg_data.merge(count_crop, on=['cropID','wave'])


agg_data = agg_data.loc[agg_data['count_hh']>=25]
agg_data.dropna(inplace=True)
crops_nation = (agg_data['cropID'].unique()).tolist()

time_mean = agg_data[['cropID','y_over_A']].groupby(by='cropID').mean()
time_mean_2 = agg_data[['cropID','y','A','m','l']].groupby(by='cropID').mean()
time_mean_2 = count_crop2.merge(time_mean_2,on='cropID')
time_mean.columns = ['avg']
time_mean.reset_index(inplace=True)
time_var =np.sqrt(agg_data[['cropID', 'y_over_A']].groupby(by='cropID').var())
time_var.columns = ['sd']
time_var.reset_index(inplace=True)

time_varmean = (pd.merge(time_mean, time_var, on='cropID')).dropna()
time_varmean['cv'] = time_varmean['sd'] / time_varmean['avg']

time_gini = pd.DataFrame(agg_data[['cropID', 'y_over_A']].groupby(by='cropID').apply(gini), columns=['gini'])
time_varmean = (pd.merge(time_varmean, time_gini, on='cropID')).dropna()


mpl.rcParams['figure.figsize'] = (10.6, 9)
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

print('===========================================')
print('FIGURE 1: CROP YIELDS VS RISK')
## PLOT BY CV: FIGURE 1 ================================== *********
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['cv'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('CV of the Yield ($/Acre)')
ax.set_ylim([0.15,1.05])
ax.set_xlim([0,1200])
#ax.set_title('Yields vs Risk (CV): Waves Variation, no Long-term Crops')
for i, txt in enumerate(time_varmean['cropID']):
    if (i == 1):
        ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,3]+0.0))
    elif (i==20):
        ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,3]+0.01))
    elif (i==16):
        ax.annotate(txt, (time_varmean.iloc[i,1]-300, time_varmean.iloc[i,3]+0.005))
    elif (i == 4) or (i == 15):
        ax.annotate(txt, (time_varmean.iloc[i,1]-170, time_varmean.iloc[i,3]-0.03))
    elif (i==14):
        ax.annotate(txt, (time_varmean.iloc[i,1]-175, time_varmean.iloc[i,3]-0.03))
    elif (i == 23):
        ax.annotate(txt, (time_varmean.iloc[i,1]-20, time_varmean.iloc[i,3]-0.03))
    elif (i == 3):
        ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,3]-0.02))
    elif (i==21):
        ax.annotate(txt, (time_varmean.iloc[i,1]-100, time_varmean.iloc[i,3]+0.01))
    elif (i==8):
        ax.annotate(txt, (time_varmean.iloc[i,1]+100, time_varmean.iloc[i,3]+0.03))
    elif (i==17):
        print('out')
    else:
        ax.annotate(txt, (time_varmean.iloc[i,1]+10, time_varmean.iloc[i,3]+0.004))
    
fig.savefig(folder_fig+'UGA_yield_vs_CV_nolong.png')
plt.show()

mpl.rcParams['figure.figsize'] = (5, 4)
### PLOT BY SD
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['sd'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: SD of the Yield')
#ax.set_title('Yields vs Risk (SD): Waves Variation, no Long-term Crops')
for i, txt in enumerate(time_varmean['cropID']):
    ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,2]), size=12)
#fig.savefig(folder_fig+'UGA_yield_vs_SD_nolong.png')
   
### PLOT BY GINI
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['gini'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: Gini of the Yield')
#ax.set_title('Yields vs Risk (Gini):  Waves Variation, no Long-term Crops')
for i, txt in enumerate(time_varmean['cropID']):
    ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,4]), size=12)
#fig.savefig(folder_fig+'UGA_yield_vs_Gini_nolong.png')
   

corr_nation_nolong = ['Nation']
corr_nation_nolong.append(time_varmean['sd'].corr(time_varmean['avg']))  ### 0.66 with all crops
corr_nation_nolong.append(time_varmean['cv'].corr(time_varmean['avg']))  ### 0.975 with all crops
corr_nation_nolong.append(time_varmean['gini'].corr(time_varmean['avg']))  ### 0.975 with all crops

### take as high productive all crops above median
crops_high_tvar_nolong = time_varmean.loc[time_varmean['avg']>np.nanmedian(time_varmean['avg']),'cropID']
### take as low productive all crops below median
crops_low_tvar_nolong = time_varmean.loc[time_varmean['avg']<np.nanmedian(time_varmean['avg']),'cropID']



print('===========================================')
print('ONLINE APPENDIX TABLE 1: OUTPUT, INPUTS, AND RISK LIST OF CROPS')

time_varmean = (pd.merge(time_varmean, time_mean_2, on='cropID')).dropna()

time_varmean.sort_values(by='avg',ascending=False ,inplace=True)
time_varmean = time_varmean[['cropID','Obs','avg','y','A','m','cv']]

time_varmean = time_varmean.sort_values(by='Obs',ascending=False)
print(time_varmean.to_latex(index=False))

#%% Yields vs Risk Variation only across time: All crops

agg_data = data[['cropID','wave','y_over_A']].groupby(by=['cropID', 'wave']).mean()
agg_data.reset_index(inplace=True)
agg_data = agg_data.merge(count_crop, on=['cropID','wave'])

agg_data = agg_data.loc[agg_data['count_hh']>=25]
agg_data.dropna(inplace=True)
crops_nation = (agg_data['cropID'].unique()).tolist()

time_mean = agg_data[['cropID','y_over_A']].groupby(by='cropID').mean()
time_mean.columns = ['avg']
time_mean.reset_index(inplace=True)
time_var =np.sqrt(agg_data[['cropID', 'y_over_A']].groupby(by='cropID').var())
time_var.columns = ['sd']
time_var.reset_index(inplace=True)

time_varmean = (pd.merge(time_mean, time_var, on='cropID')).dropna()
time_varmean['cv'] = time_varmean['sd'] / time_varmean['avg']

time_gini = pd.DataFrame(agg_data[['cropID', 'y_over_A']].groupby(by='cropID').apply(gini), columns=['gini'])
time_varmean = (pd.merge(time_varmean, time_gini, on='cropID')).dropna()


## PLOT BY CV
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['cv'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: CV of the Yield')
#ax.set_title('Crops Yields vs Risk (CV): Nationwide')
for i, txt in enumerate(time_varmean['cropID']):
    ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,3]), size=12)
#fig.savefig(folder_fig+'UGA_yield_vs_CV.png')
    

### PLOT BY SD
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['sd'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: SD of the Yield')
ax.set_title('Crops Yields vs Risk (SD): Nationwide')
for i, txt in enumerate(time_varmean['cropID']):
    ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,2]), size=12)
#fig.savefig(folder_fig+'UGA_yield_vs_SD.png')


### PLOT BY GINI
fig, ax = plt.subplots()
ax.scatter(time_varmean['avg'], time_varmean['gini'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: Gini of the Yield')
#ax.set_title('Crops Yields vs Risk (Gini): Nationwide')
for i, txt in enumerate(time_varmean['cropID']):
    ax.annotate(txt, (time_varmean.iloc[i,1], time_varmean.iloc[i,4]), size=12)
#fig.savefig(folder_fig+'UGA_yield_vs_Gini.png')
   
corr_nation = ['Nation']
corr_nation.append(time_varmean['sd'].corr(time_varmean['avg']))  
corr_nation.append(time_varmean['cv'].corr(time_varmean['avg']))  
corr_nation.append(time_varmean['gini'].corr(time_varmean['avg']))  



### take as high productive all crops above median
crops_high_tvar = time_varmean.loc[time_varmean['avg']>np.nanmedian(time_varmean['avg']),'cropID']

### take as low productive all crops below median
crops_low_tvar = time_varmean.loc[time_varmean['avg']<np.nanmedian(time_varmean['avg']),'cropID']

print('===========================================')
print('TABLE 1: CORRELATIONS CROP YIELDS AND CROP RISK.')
print('Correlations obs. grouped at wave level, all crops. columns 2 and 3-6, rows 3-6, table 1')
#### Variation across waves -----------------
agg_data = data[['cropID','wave', 'region','y_over_A', 'kg_over_A']].groupby(by=['cropID', 'wave', 'region']).mean()
agg_data.reset_index(inplace=True)
agg_data = agg_data.merge(count_cropregion, on=['cropID','wave', 'region'])


## at least 10 households in each crop, region, season, combination
agg_data = agg_data.loc[agg_data['count']>=10]
agg_data.dropna(inplace=True)

region_corr_cv = []
region_corr_sd = []
region_corr_gini = []
region_corr_cv_nolong = []
region_corr_sd_nolong = []
region_corr_gini_nolong = []

crops_region = []
for region in np.array([1.0,3.0,4.0,2.0]):
    #print('REGION NUMBER:'+str(region))
    if region == 1.0:
        region_label = 'Central'
    if region == 2.0:
        region_label = 'Eastern'
    elif region == 3.0:
        region_label = 'Northern'
    elif region == 4.0:
        region_label = 'Western'
    
                
    data_region = agg_data.loc[agg_data['region']==region]
    crops_region.append((data_region['cropID'].unique()).tolist())
    
    time_mean = data_region[['cropID','y_over_A']].groupby(by='cropID').mean()
    time_mean.columns = ['avg']
    time_mean.reset_index(inplace=True)
    time_var =np.sqrt(data_region[['cropID', 'y_over_A']].groupby(by='cropID').var())
    time_var.columns = ['sd']
    time_var.reset_index(inplace=True)
    time_varmean = (pd.merge(time_mean, time_var, on='cropID')).dropna()
    time_varmean['cv'] = time_varmean['sd'] / time_varmean['avg']
    time_gini = pd.DataFrame(data_region[['cropID', 'y_over_A']].groupby(by='cropID').apply(gini), columns=['gini'])
    time_varmean = (pd.merge(time_varmean, time_gini, on='cropID')).dropna()    
    region_corr_cv.append(time_varmean['cv'].corr(time_varmean['avg']))   
    region_corr_sd.append(time_varmean['sd'].corr(time_varmean['avg'])) 
    region_corr_gini.append(time_varmean['sd'].corr(time_varmean['gini']))    
    
    data_region_nolong = data_region.loc[~data_region['cropID'].isin(crops_long)]
    
    time_mean2 = data_region_nolong[['cropID','y_over_A']].groupby(by='cropID').mean()
    time_mean2.columns = ['avg']
    time_mean2.reset_index(inplace=True)
    time_var2 =np.sqrt(data_region_nolong[['cropID', 'y_over_A']].groupby(by='cropID').var())
    time_var2.columns = ['sd']
    time_var2.reset_index(inplace=True)
    time_varmean2 = (pd.merge(time_mean2, time_var2, on='cropID')).dropna()
    time_varmean2['cv'] = time_varmean2['sd'] / time_varmean2['avg']
    time_gini2 = pd.DataFrame(data_region_nolong[['cropID', 'y_over_A']].groupby(by='cropID').apply(gini), columns=['gini'])
    time_varmean2 = (pd.merge(time_varmean2, time_gini2, on='cropID')).dropna()
    
    region_corr_cv_nolong.append(time_varmean2['cv'].corr(time_varmean2['avg']))   ### 0.66 with all crops
    region_corr_sd_nolong.append(time_varmean2['sd'].corr(time_varmean2['avg'])) ### 0.975 with all crops
    region_corr_gini_nolong.append(time_varmean2['sd'].corr(time_varmean2['gini']))
    
    
    fig, ax = plt.subplots()
    ax.scatter(time_varmean2['avg'], time_varmean2['sd'],edgecolors='face')
    ax.set_xlabel('Average Yield ($/Acre)')
    ax.set_ylabel('Risk: SD of the Yield')
    #ax.set_title('Crops Yields vs Risk (SD): '+region_label+' Uganda, No Long-term Crops')
    for i, txt in enumerate(time_varmean2['cropID']):
         x = time_varmean2.loc[time_varmean2['cropID']==txt,'avg']
         y = time_varmean2.loc[time_varmean2['cropID']==txt,'sd']
         #ax.annotate(txt, (x,y))
         ax.annotate(txt, (time_varmean2.iloc[i,1], time_varmean2.iloc[i,2]), size=12)
    #fig.savefig(folder_fig+region_label+'yield_vs_SD_nolong.png')
    
    
    fig, ax = plt.subplots()
    ax.scatter(time_varmean2['avg'], time_varmean2['cv'],edgecolors='face')
    ax.set_xlabel('Average Yield ($/Acre)')
    ax.set_ylabel('Risk: CV of the Yield')
    #ax.set_title('Crops Yields vs Risk (CV): '+region_label+' Uganda, No Long-term Crops')
    for i, txt in enumerate(time_varmean2['cropID']):
        ax.annotate(txt, (time_varmean2.iloc[i,1], time_varmean2.iloc[i,3]), size=12)
    #fig.savefig(folder_fig+region_label+'yield_vs_CV_nolong.png')
    
    
    fig, ax = plt.subplots()
    ax.scatter(time_varmean2['avg'], time_varmean2['gini'],edgecolors='face')
    ax.set_xlabel('Average Yield ($/Acre)')
    ax.set_ylabel('Risk: CV of the Yield')
    #ax.set_title('Crops Yields vs Risk (Gini): '+region_label+' Uganda, No Long-term Crops')
    for i, txt in enumerate(time_varmean2['cropID']):
        ax.annotate(txt, (time_varmean2.iloc[i,1], time_varmean2.iloc[i,4]), size=12)
    #fig.savefig(folder_fig+region_label+'yield_vs_gini_nolong.png')
    

region_corr_sd[:-1]

regions_corr_dict = [('Region', ['Central','Eastern', 'Northern', 'Western']),
         ('SD', region_corr_sd),
         ('CV', region_corr_cv),
         ('Gini', region_corr_gini),]

regions_corr = pd.DataFrame.from_dict(dict((regions_corr_dict)))
del region_corr_cv, region_corr_sd, region_corr_gini 
#regions_corr.drop(4, inplace=True)
print('Correlations nationwide :')
print(str(corr_nation))
print((regions_corr.transpose()).to_latex())



regions_corr_dict_nolong = [('Region', ['Central','Eastern', 'Northern', 'Western']),
         ('SD', region_corr_sd_nolong),
         ('CV', region_corr_cv_nolong),
         ('Gini', region_corr_gini_nolong),]

regions_corr_nolong = pd.DataFrame.from_dict(dict((regions_corr_dict_nolong)))

del region_corr_cv_nolong, region_corr_sd_nolong, region_corr_gini_nolong 

print('Correlations nationwide no long-term crops. Columns 2 and 3-6, rows 1-3, Table 1:')
print(str(corr_nation_nolong))
print((regions_corr_nolong.transpose()).to_latex())



print('===========================================')
print('Risk vs Yields Crops: Variation across time and households all crops. columns 1, rows 4-6) table 1')

# Summary table crops (in online appendix)
# computation crop shares consumption, selling, (not used)

count_crop = data.groupby(by=['cropID','wave'])[['y']].count()
count_crop.columns = ['count_hh']
count_crop.reset_index(inplace=True)

agg_data = data[['cropID','wave','y_over_A']].groupby(by=['cropID', 'wave']).mean()
agg_data.reset_index(inplace=True)
agg_data = agg_data.merge(count_crop, on=['cropID','wave'])

agg_data = agg_data.loc[agg_data['count_hh']>=25]
agg_data.dropna(inplace=True)
list_crops = (agg_data['cropID'].unique()).tolist()

crop_summary = []
yield_mean = []
yield_sd = []
yield_cv = []

yield_mean_agg = []
yield_sd_agg = []
yield_cv_agg = []
yield_gini_agg = []
y_agg= []
A_agg = []
m_agg = []
l_agg = []

inputs = []
inputs_list = []

data['interm_value_p_sell_reg'] = data[["food_prod_value_p_sell_reg","animal_value_p_sell_reg"]].sum(axis=1)


data['consuming'] = data['cons_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['intermediates'] = data['interm_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['storing'] = data['stored_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['gifting'] = data['gift_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']

mean_selling = []
mean_interm = []
mean_cons = []
mean_stor = []
mean_gift = []
n_list = []
for item in list_crops:
    
     #Get data by crop
     data_crop=data.loc[data['cropID']==item, ['y','y_over_A','A','m','l']]
     n = len(data_crop)
     n_list.append(n)
     data_xharvest = data.loc[data['cropID']==item, ['y_over_A','wave']]
     data_inputs = data.loc[data['cropID']==item, ['y','A','l','m','wave']]
     data_shares = data.loc[data['cropID']==item, ['consuming','intermediates','storing','gifting']]  
     mean_shares = np.mean(data_shares)
     
     
     mean_interm.append(np.mean(data_shares['intermediates']))
     mean_cons.append(np.mean(data_shares['consuming']))
     mean_stor.append(np.mean(data_shares['storing']))
     mean_gift.append(np.mean(data_shares['gifting']))
   
     #Compute aggregate statistics
     summary = data_crop.describe()
     mean_agg = np.mean(data_crop['y_over_A'])
     y_agg_temp =data_crop['y'].mean()
     A_agg_temp = data_crop['A'].mean()
     m_agg_temp = data_crop['m'].mean()
     l_agg_temp = data_crop['l'].mean()
     
     sd_agg = np.var(data_crop['y_over_A'])
     gini_agg = gini(np.array(data_crop['y_over_A'].dropna()))
     cv_agg = sd_agg / mean_agg    
     # Append them to list
     yield_mean_agg.append(mean_agg)
     yield_sd_agg.append(sd_agg)
     yield_cv_agg.append(cv_agg)
     yield_gini_agg.append(gini_agg)
     y_agg.append(y_agg_temp)
     A_agg.append(A_agg_temp)
     m_agg.append(m_agg_temp)
     l_agg.append(l_agg_temp)
     
     #Compute statistics per survey wave
     mean = data_xharvest.groupby(by=["wave"]).mean()  
     sd = data_xharvest.groupby(by=["wave"]).std() 
     inputs = data_inputs.groupby(by=["wave"]).mean()
     cv =  sd/mean 
     
     #Append them to list
     crop_summary.append(summary)
     yield_mean.append(mean)
     yield_sd.append(sd)
     inputs_list.append(inputs)
     yield_cv.append(cv)


### Productivity and Risk aggregating waves
data_crops = [('Crop', list_crops),
          ('Obs', n_list), 
         ('Mean', yield_mean_agg),
         ('SD', yield_sd_agg),
         ('CV', yield_cv_agg),
         ('Gini', yield_gini_agg),
         ]
data_crops_agg = pd.DataFrame.from_dict(dict((data_crops)))

### take as high productive all crops above median
data_high = data_crops_agg.loc[data_crops_agg['Mean']>np.nanmedian(data_crops_agg['Mean']),'Crop']
### take as low productive all crops below median
data_low = data_crops_agg.loc[data_crops_agg['Mean']<np.nanmedian(data_crops_agg['Mean']),'Crop']



## ---------

data_shares = [('Crop', list_crops),
         ('Obs', n_list),
         ('Cons', mean_cons),
         ('interm', mean_interm),
         ('Store', mean_stor),
         ('gift', mean_gift),
         ('Mean', yield_mean_agg),
         ('CV', yield_cv_agg),
         ]
data_shares_output = pd.DataFrame.from_dict(dict((data_shares)))

data_shares_output = data_shares_output.sort_values(by='Obs')
# shares by crops
#print(data_shares_output.to_latex(index=False))
#print('interesting table. shows that both most high and low crops are for own-consumption which helps argument paper.')

### Scatter plot-----------------------

fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['CV'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: CV of the Yield')
#ax.set_title('Crops Yields vs Risk (CV): Variation across Farmers and Waves')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,4]), size=12)
#fig.savefig(folder_fig+'yield_vs_CV.png')
   


## SD
fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['SD'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: SD of the Yield')
#ax.set_title('Crops Yields vs Risk (SD): Variation across Farmers and Waves')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,3]), size=12)
#fig.savefig(folder_fig+'yield_vs_SD.png')
   

## Gini
fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['Gini'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: Gini of the Yield')
#ax.set_title('Crops Yields vs Risk (Gini): Variation across Farmers and Waves')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,5]), size=12)
#fig.savefig(folder_fig+'yield_vs_Gini.png')
   


corr_mean_sd = data_crops_agg['Mean'].corr(data_crops_agg['SD'])
corr_mean_cv = data_crops_agg['Mean'].corr(data_crops_agg['CV'])
corr_mean_gini = data_crops_agg['Mean'].corr(data_crops_agg['Gini'])

print([corr_mean_sd, corr_mean_cv, corr_mean_gini])

print('Risk vs Yields Crops: Variation across time and households, only no-long term crops. columns 1, rows 1-3) table 1')




pd.options.display.float_format = '{:,.2f}'.format
crops_long = ['Avocado','Cocoa', 'Mango', 'Coffee All', 'Paw Paw', 'Tobacco', 'Passion Fruit', 'Oranges', 'Vanilla', 'Jackfruit']
### TAKE ONLY CROPS THAT GROW WITHIN A SEASON OR YEAR

data_nolong = data.loc[~data['cropID'].isin(crops_long)]

count_crop = data_nolong.groupby(by=['cropID','wave'])[['y']].count()
count_crop.columns = ['count_hh']
count_crop.reset_index(inplace=True)

agg_data = data_nolong[['cropID','wave','y_over_A']].groupby(by=['cropID', 'wave']).mean()
agg_data.reset_index(inplace=True)
agg_data = agg_data.merge(count_crop, on=['cropID','wave'])

agg_data = agg_data.loc[agg_data['count_hh']>=25]
agg_data.dropna(inplace=True)
list_crops = (agg_data['cropID'].unique()).tolist()

crop_summary = []
yield_mean = []
yield_sd = []
yield_cv = []

yield_mean_agg = []
yield_sd_agg = []
yield_cv_agg = []
yield_gini_agg = []
y_agg= []
A_agg = []
m_agg = []
l_agg = []

inputs = []
inputs_list = []


data['consuming'] = data['cons_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['intermediates'] = data['interm_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['storing'] = data['stored_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']
data['gifting'] = data['gift_value_p_sell_reg'].fillna(0)/data['total2_value_p_sell_reg']


mean_selling = []
mean_interm = []
mean_cons = []
mean_stor = []
mean_gift = []
n_list = []
for item in list_crops:
    
     #Get data by crop
     data_crop=data.loc[data['cropID']==item, ['y','y_over_A','A','m','l']]
     n = len(data_crop)
     n_list.append(n)
     data_xharvest = data.loc[data['cropID']==item, ['y_over_A','wave']]
     data_inputs = data.loc[data['cropID']==item, ['y','A','l','m','wave']]
     data_shares = data.loc[data['cropID']==item, ['consuming','intermediates','storing','gifting']]
     
     mean_shares = np.mean(data_shares)
     
     
     #Compute aggregate statistics
     summary = data_crop.describe()
     mean_agg = np.mean(data_crop['y_over_A'])
     y_agg_temp =data_crop['y'].mean()
     A_agg_temp = data_crop['A'].mean()
     m_agg_temp = data_crop['m'].mean()
     l_agg_temp = data_crop['l'].mean()
     
     sd_agg = np.var(data_crop['y_over_A'])
     gini_agg = gini(np.array(data_crop['y_over_A'].dropna()))
     cv_agg = sd_agg / mean_agg    
     # Append them to list
     yield_mean_agg.append(mean_agg)
     yield_sd_agg.append(sd_agg)
     yield_cv_agg.append(cv_agg)
     yield_gini_agg.append(gini_agg)
     y_agg.append(y_agg_temp)
     A_agg.append(A_agg_temp)
     m_agg.append(m_agg_temp)
     l_agg.append(l_agg_temp)
     
     #Compute statistics per survey wave
     mean = data_xharvest.groupby(by=["wave"]).mean()  
     sd = data_xharvest.groupby(by=["wave"]).std() 
     inputs = data_inputs.groupby(by=["wave"]).mean()
     cv =  sd/mean 
     
     #Append them to list
     crop_summary.append(summary)
     yield_mean.append(mean)
     yield_sd.append(sd)
     inputs_list.append(inputs)
     yield_cv.append(cv)


### Productivity and Risk aggregating waves
data_crops = [('Crop', list_crops),
          ('Obs', n_list), 
         ('Mean', yield_mean_agg),
         ('SD', yield_sd_agg),
         ('CV', yield_cv_agg),
         ('Gini', yield_gini_agg),
         ]
data_crops_agg = pd.DataFrame.from_dict(dict((data_crops)))
data_crops_agg = data_crops_agg.sort_values(by='Obs', ascending=False)



### Scatter plot-----------------------
### CV
fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['CV'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: CV of the Yield')
#ax.set_title('Crops Yields vs Risk (CV): Variation across Farmers and Waves, No Long-term Crops')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,4]))
#fig.savefig(folder_fig+'yield_vs_CV_nolong.png')
   
## SD
fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['SD'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: SD of the Yield')
#ax.set_title('Yields vs Risk (SD): Variation across Farmers and Waves, No Long-term Crops')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,3]))
#fig.savefig(folder_fig+'yield_vs_SD_nolong.png')
   

## Gini
fig, ax = plt.subplots()
ax.scatter(data_crops_agg['Mean'], data_crops_agg['Gini'],edgecolors='face')
ax.set_xlabel('Average Yield ($/Acre)')
ax.set_ylabel('Risk: Gini of the Yield')
#ax.set_title('Crops Yields vs Risk (Gini): variation across Farmers and Waves, No Long-term Crops')
for i, txt in enumerate(data_crops_agg['Crop']):
    ax.annotate(txt, (data_crops_agg.iloc[i,2], data_crops_agg.iloc[i,5]))
#fig.savefig(folder_fig+'yield_vs_Gini_nolong.png')
   

corr_mean_sd = data_crops_agg['Mean'].corr(data_crops_agg['SD'])
corr_mean_cv = data_crops_agg['Mean'].corr(data_crops_agg['CV'])
corr_mean_gini = data_crops_agg['Mean'].corr(data_crops_agg['Gini'])

corr_nation_hht_nolong = [corr_mean_sd, corr_mean_cv,corr_mean_gini]

print(corr_nation_hht_nolong )



