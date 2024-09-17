# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:53:01 2019

@author: rodri
"""
# =============================================================================
# Comparing Model vs Data outcomes
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
print(os.getcwd())

dirct  = Path('Master_quantmodel.py').resolve().parent.parent # to run the file alone (not through master) remove 1 .parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_dirct = dirct+'/quant model/'
os.chdir(my_dirct)
    
    
from model_class import HH_model
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col


# Display options
pd.options.display.float_format = '{:,.4f}'.format
from matplotlib.ticker import FormatStrFormatter
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

#Folders
folder = dirct+'/Results/'
folder2 = dirct+'/data/panel/'
folder_fig =  dirct+'/Results/figures/model vs data/'

save = True


def gini(array):
    # from: https://github.com/oliviaguest/gini
    #http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm 
    array = np.array(array)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array += np.amin(array) #non-negative
    array += 0.0000001 #non-0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) 
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 



        
cp = HH_model()


        
#### Get model outcomes        
data_sim = pd.read_csv(folder+'IM_solution_10periods.csv.gz')
names_list = ['High input','Low Input','Output','Assets','Consumption']
data_sim[['X']].describe(percentiles=[0.1,0.05,0.5,0.75,0.9,0.95,0.99])
## For stationary state (last period)
data_ss = data_sim.loc[data_sim['t']==np.max(data_sim['t']),]

data_t = data_sim.groupby(by='t').mean()
data_var = data_sim.groupby(by='t').var()

mean_list = ['y','mh','ml','a','c' ]
plot_agg=True
if plot_agg == True:
### Plot means across time
    fig, ax = plt.subplots()
    for i,mean in enumerate(mean_list):
        ax.plot(range(0,10), data_t[mean], label=mean_list[i])
        ax.legend()
        ax.set_xlabel('Time')
        #ax.set_title('Average across time')
        #ax.set_ylim((-200,1000))
    fig.savefig(folder+'figures/model/economy/cvg_distr_means.png')                  
    plt.show()     

### Plot variance across time
    fig, ax = plt.subplots()
    for i,mean in enumerate(mean_list):
        ax.plot(range(0,10), data_var[mean], label=mean_list[i])
        ax.legend()
        ax.set_xlabel('Time')
        #ax.set_title('Variance across time')  
    fig.savefig(folder+'figures/model/economy/cvg_distr_variance.png')                
    plt.show()     


#%%
### States distribution in stationarity  =============
data_ss['y_agr'] = data_ss[['yh','yl']].sum(axis=1) 

### Marginal distributions: CIW + inputs, agric production'
pd.options.display.float_format = '{:,.2f}'.format
desc_stats = data_ss[['X', 'z', 'yna', 'y', 'yh', 'yl', 'mh', 'ml', 'a', 'c']].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

desc_stats_filtered = desc_stats.drop('count')

print(' === TABLE 1 IN THE ONLINE APPENDIX =====')
print(desc_stats_filtered.to_latex())


share_yh = np.mean(data_ss['yh']/data_ss['y_agr'])
share_yhl = np.mean(data_ss['yh'])/np.mean(data_ss['yl'])
check = pd.value_counts(data_ss['yna'])/len(data_ss)


### To logs
data_sim['lny_h'] = np.log(data_sim['yh'])
data_sim['lny_l'] = np.log(data_sim['yl'])
data_sim['lny'] = np.log(data_sim['y'])
data_sim['lnc'] = np.log(data_sim['c'])
data_sim['lna'] = np.log(data_sim['a'])



#TARGETED MOMENTS ===============================

# Averages
means = data_ss[['yh','yl','y','a']].mean()
m_means = data_ss[['mh','ml']].mean()

# Risk
timevar = data_sim.groupby(by='hh')[['lny_h', 'lny_l']].cov()
timevar.reset_index(inplace=True)
blu = timevar.groupby(by='level_1').mean()
sd_yh = np.sqrt(blu.iloc[0,1])
sd_yl = np.sqrt(blu.iloc[1,2])

cov_yhyl = blu.iloc[1,1]/(sd_yh*sd_yl)
sigmas = [ blu.iloc[0,1], blu.iloc[1,2], cov_yhyl]
vol_i = (data_sim.groupby(by='hh')[['lny','lnc']].var()).mean(axis=0)

# Ineq
gini_y = gini(data_ss['y'])
gini_ya = gini(data_ss['y_agr'])
gini_c = gini(data_ss['c'])

tmomt_labels = ['yh', 'yl', 'y','a','vol_yh','vol_yl','corr_yhyl', 'gini_agric']
ntmomt_labels = ['mh','ml', 'vol_i', 'vol_c']

momt = []
ntmomt = []

momt.extend(means)
momt.extend(sigmas)

momt.append(gini_ya)

ntmomt.extend(m_means)
ntmomt.extend(vol_i)



pd.options.display.float_format = '{:,.4f}'.format

tmomt_data = [651.12, 353.39,  1506, 1858, 1.052, 0.722, 0.115, 0.62]
ntmomt_data = [40.75, 36.46, 0.600,0.117]

df_momt = pd.DataFrame({'Moment':pd.Series(tmomt_labels),'Model':pd.Series(momt),'Data':pd.Series(tmomt_data)})
print('CALIBRATION: MODEL VS DATA MOMENTS         ')
print('TABLE 4 IN THE PAPER (ON TARGET MOMENTS)')
print('             ')
print((df_momt.applymap(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else x)).to_latex(index=False))







#%% Crops share 

#crops shares statistics
yh_share = np.mean(data_ss['yh'])/np.mean(data_ss['y_agr'])
print('Model predicts share of high crops on total agric output of',100*yh_share)

yh_share_d = 651.12/(651.12+353.39)

print('data share of high crops on total agric output of',100*np.mean(yh_share_d))

# crops share along wealth: model vs data
percentiles = np.linspace(0,100,21)

n_p = len(percentiles)
data_x = data_ss.groupby(pd.cut(data_ss.X, np.percentile(data_ss['X'].dropna(), percentiles), include_lowest=True)).mean()

data_x['yh_over_y'] = (data_x['yh'].divide(data_x['y_agr'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
data_x['yl_over_y'] = (data_x['yl'].divide(data_x['y_agr'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)


data_cwi = pd.read_csv(folder2+'panelrural_highlowcrops.csv')

data_y = data_cwi.groupby(pd.cut(data_cwi.wtotal, np.percentile(data_cwi['wtotal'].dropna(), percentiles), include_lowest=True)).mean()
 
data_y['y'] = (data_y['y_h'].fillna(0) + data_y['y_l'].fillna(0)).replace(0,np.nan)
data_y['yh_over_y'] = (data_y['y_h'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)
data_y['yl_over_y'] = (data_y['y_l'].divide(data_y['y'].replace([np.inf,0],np.nan))).replace([np.inf,0],np.nan)

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
 

print('     ')
print('===========================================')
print('FIGURE 4 HIGH CROPS SHARE ALONG WEALT: MODEL VS DATA')
fig, ax = plt.subplots()
ax.plot(percentiles[1:n_p], data_y['yh_over_y'], label = 'high crops data', color=colormap[0] )
ax.plot(percentiles[1:n_p], data_y['yl_over_y'], label = 'Low crops data', color=colormap[1])
ax.plot(percentiles[1:n_p], data_x['yh_over_y'], label = 'high crops model', color=colormap[0], linestyle='dashed')
ax.plot(percentiles[1:n_p], data_x['yl_over_y'], label = 'Low crops model', color=colormap[1], linestyle='dashed')

lines = ax.get_lines()
legend1 = plt.legend([lines[i] for i in [0,1]], ["High Crops", "Low Crops"], loc=1)
legend2 = plt.legend([lines[i] for i in [0,2]], ["Data", "Model"], loc=2, labelcolor='black')
ax.add_artist(legend1)
ax.add_artist(legend2)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(percentiles[1:n_p])
ax.set_xlabel('Wealth Distribution')
#ax.set_title('Low vs High Crops Output Shares along Wealth')
ax.set_ylabel('Crop Output over Total (Share)')
ax.set_ylim((0.2,0.8))


if save==True:
    fig.savefig(folder_fig+'crops_along_wealth_MvsD.png')
plt.plot()
plt.show()


#%% Consumption, Income, and Wealth Distributions

## data CIW
data_cwi = pd.read_csv(folder2+'panel_rural_UGA.csv')
data_cwi.reset_index(inplace=True)
percentiles = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99]

pd.options.display.float_format = '{:,.2f}'.format

# 2011-12 wave
data_cwi11 =  data_cwi.loc[data_cwi['wave']=='2011-2012']


### Income and consumption distributions: model vs data

def plot_distribution_waves(state_model, state_data, state_name, bw_kernel=0.2, x_lim=[0,40000], x_lim_true=False, save=False, folder=folder):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(state_data, label='Data',kde=True, kde_kws={"lw": 3,'bw':bw_kernel,'color':'black', 'linestyle':'-'},  hist=False)
    sns.distplot(state_model, label='Model', kde=True, kde_kws={ "lw": 3, 'bw':bw_kernel,'color':'red', 'linestyle':'--'},  hist=False)
    
    #plt.title('Distribution of '+state_name)
    if x_lim_true==True:
        plt.xlim(x_lim)
       #plt.ylim([0,0.7])
    plt.xlabel('Log of '+state_name)
    plt.ylabel("Density")
    plt.legend()
    if save==True:
        fig.savefig(folder_fig+'Distr_'+state_name+'_MvsD11.png')
    return plt.show()


print('     ')
print('===========================================')
print('FIGURE 5: CIW DISTRIBUTIONS, MODEL VS DATA')

## Income
plot_distribution_waves(np.log(data_ss['y'])-np.log(np.mean(data_ss['y'])),
                        np.log(data_cwi11['inctotal'])-np.log(np.mean(data_cwi11['inctotal'])),
                        'income', save=save, x_lim=((-6,6)), x_lim_true=True) 
 

## Consumption
c_mean11 = np.mean(data_cwi11.loc[data_cwi11['ctotal']>0.01,'ctotal'].dropna())

plot_distribution_waves(np.log(data_ss['c'])-np.log(np.mean(data_ss['c'])),
                        np.log(data_cwi11['ctotal'])-np.log(c_mean11),
                        'Consumption', save=save, x_lim=((-3,3)), x_lim_true=True)



w_mean11 = np.mean(data_cwi11.loc[data_cwi11['liquid_w']>0.01,'liquid_w'].dropna())

# Wealth
plot_distribution_waves(np.log(data_ss['a'])-np.log(np.mean(data_ss['a'])),
                        np.log(data_cwi11['liquid_w'])-np.log(c_mean11),
                        'Wealth', save=save, x_lim=((-6,5)), x_lim_true=True)


### CIW shares: model vs data

# data
gini_inc = gini(data_cwi11['inctotal'])
gini_w = gini(data_cwi11['liquid_w'].dropna())
gini_c = gini(data_cwi11['ctotal'])

income_shares = []
w_shares = []
c_shares = []

# model
data_ss['y'], data_ss['a'], data_ss['c']

mgini_inc = gini(data_ss['y'])
mgini_w = gini(data_ss['a'])
mgini_c = gini(data_ss['c'])

minc_shares = []
mw_shares = []
mc_shares = []


for p in [1, 5, 10, 25, 50]:
    
    ## DATA ----------------------
    #income
    data_p = data_cwi11.loc[ data_cwi11['inctotal'] <= np.percentile(data_cwi11['inctotal'].dropna(), p)]
    gain = (np.sum(data_p['inctotal']) / np.sum(data_cwi11['inctotal']))*100
    income_shares.append(gain)
    #wealth
    data_w = data_cwi11.loc[ data_cwi11['liquid_w'] <= np.percentile(data_cwi11['liquid_w'].dropna(), p)]
    gain_w = (np.sum(data_w['liquid_w']) / np.sum(data_cwi11['liquid_w']))*100
    w_shares.append(gain_w)
    #consumption
    data_c = data_cwi11.loc[ data_cwi11['ctotal'] <= np.percentile(data_cwi11['ctotal'].dropna(), p)]
    gain_c = (np.sum(data_c['ctotal']) / np.sum(data_cwi11['ctotal']))*100
    c_shares.append(gain_c)
    
    ## MODEL -------------------
    data_mi = data_ss.loc[ data_ss['y'] <= np.percentile(data_ss['y'].dropna(), p)]
    gain_mi = (np.sum(data_mi['y']) / np.sum(data_ss['y']))*100
    minc_shares.append(gain_mi)
    #wealth
    data_mw = data_ss.loc[ data_ss['a'] <= np.percentile(data_ss['a'].dropna(), p)]
    gain_mw = (np.sum(data_mw['a']) / np.sum(data_ss['a']))*100
    mw_shares.append(gain_mw)
    #consumption
    data_mc = data_ss.loc[ data_ss['c'] <= np.percentile(data_ss['c'].dropna(), p)]
    gain_mc = (np.sum(data_mc['c']) / np.sum(data_ss['c']))*100
    mc_shares.append(gain_mc)
    
    
    
for p in [50, 75, 90, 95, 99]:
    ## DATA ----------------------
    data_p = data_cwi11.loc[ data_cwi11['inctotal'] >= np.percentile(data_cwi11['inctotal'].dropna(), p)]
    gain = (np.sum(data_p['inctotal']) / np.sum(data_cwi11['inctotal']))*100
    income_shares.append(gain)
    data_w = data_cwi11.loc[ data_cwi11['liquid_w'] >= np.percentile(data_cwi11['liquid_w'].dropna(), p)]
    gain_w = (np.sum(data_w['liquid_w']) / np.sum(data_cwi11['liquid_w']))*100
    w_shares.append(gain_w)
    #consumption
    data_c = data_cwi11.loc[ data_cwi11['ctotal'] >= np.percentile(data_cwi11['ctotal'].dropna(), p)]
    gain_c = (np.sum(data_c['ctotal']) / np.sum(data_cwi11['ctotal']))*100
    c_shares.append(gain_c)
    
    ## MODEL -------------------
    data_mi = data_ss.loc[ data_ss['y'] >= np.percentile(data_ss['y'].dropna(), p)]
    gain_mi = (np.sum(data_mi['y']) / np.sum(data_ss['y']))*100
    minc_shares.append(gain_mi)
    #wealth
    data_mw = data_ss.loc[ data_ss['a'] >= np.percentile(data_ss['a'].dropna(), p)]
    gain_mw = (np.sum(data_mw['a']) / np.sum(data_ss['a']))*100
    mw_shares.append(gain_mw)
    #consumption
    data_mc = data_ss.loc[ data_ss['c'] >= np.percentile(data_ss['c'].dropna(), p)]
    gain_mc = (np.sum(data_mc['c']) / np.sum(data_ss['c']))*100
    mc_shares.append(gain_mc)


list_percentiles = [1, 5, 10, 25, 50, 50, 75, 90, 95, 99]


inequality_table = (pd.DataFrame({'Percentiles': list_percentiles,'Income':income_shares,'Wealth':w_shares, 'Consumption':c_shares, 'Income (M)':minc_shares,'Wealth (M)':mw_shares, 'Consumption (M)':mc_shares})).transpose()
inequality_table.columns = inequality_table.iloc[0]


print('     ')
print('===========================================')
print('TABLE 5 IN PAPER: CIW INEQUALITY: MODEL VS. DATA')
print((inequality_table.applymap(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)).to_latex())

pd.options.display.float_format = '{:.2f}'.format


## Income and Consumption Volatility
vol_i = (data_sim.groupby(by='hh')[['lny','lnc']].var()).mean(axis=0)
vol_labels = [ 'vol_i', 'vol_c']
vol_i_data = [0.600, 0.117]


df_momt = pd.DataFrame({
    'Moment': vol_labels,
    'Model': vol_i.values,  # Use .values to extract data from Series as a list
    'Data': vol_i_data
})

print('       ')
print('TABLE 13 IN APPENDIX: INCOME AND CONSUMPTION VOLATILITY: MODEL VS. DATA')
print('             ')
print((df_momt.applymap(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else x)).to_latex(index=False))



print(os.getcwd())
