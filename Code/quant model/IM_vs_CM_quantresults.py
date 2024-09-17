
# =============================================================================
#  QUANTITATIVE RESULTS: THE GAINS OF COMPLETING FINANCIAL MARKETS
    # BENCHMARK VS COMPLETE MARKETS
# ==========================================================================


import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_data.py').resolve().parent.parent # to run the file alone (not through master) remove 1 .parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_dirct = dirct+'/quant model/'   
os.chdir(my_dirct)
folder = dirct+'/Results/'   

from model_class import HH_model, gini
pd.options.display.float_format = '{:,.2f}'.format


cp = HH_model()
A,B = cp.A, cp.B
alpha, gamma, rho, bbeta = cp.alpha, cp.gamma, cp.rho, cp.bbeta
r, p = cp.r, cp.p
SIGMA = cp.SIGMA
sig_theta = cp.sig_theta
sig_eps = cp.sig_eps
sig_z = cp.sig_z
N = cp.N
p_z = cp.p_z
### permanent productivity grid -----
N_z, z_grid= cp.N_z, cp.z_grid
p=cp.p

mh_star = (p/(z_grid*A*alpha))**(1/(alpha-1))
ml_star =  (p/(z_grid*B*gamma))**(1/(gamma-1))
# productivity ------------------
p = cp.p

#### Get Benchmarck economy (IM) outcomes------------------------     
data_sim =  pd.read_csv(folder+'IM_solution_10periods.csv.gz')

data_sim['y_agr'] = data_sim[['yh','yl']].sum(axis=1) 
data_sim['lnya'] = np.log(data_sim['y_agr'])

## For stationary state (last period)
bench_ss = data_sim.loc[data_sim['t']==np.max(data_sim['t']),] 
bench_ss['m'] = bench_ss[['mh','ml']].sum(axis=1) 

print(bench_ss[['mh','ml']].mean())


### Get CM outcomes---------------------------------
data_sp = pd.read_csv(folder+'CM_solution_10periods.csv.gz')
data_sp['y_agr'] = data_sp[['yh','yl']].sum(axis=1) 
data_sp['lnya'] = np.log(data_sp['y_agr'])
data_sp['lny'] = np.log(data_sp['y'])
data_sp['lnc'] = np.log(data_sp['c'])
data_sp['lny_h'] = np.log(data_sp['yh'])
data_sp['lny_l'] = np.log(data_sp['yl'])

# Stationary state
data_1t = data_sp.loc[data_sp['t']==np.max(data_sp['t']),]
data_1t['m'] = data_1t[['mh','ml']].sum(axis=1) 


# AGRICULTURAL PRODUCTIVITY ===========================================
m1_mom = np.mean(bench_ss['mh'])
m2_mom = np.mean(bench_ss['ml'])
y_mom = np.mean(bench_ss['y_agr'])

### AGRICULTURAL OUTPUT AND INPUT GAINS: TOTAL AND ACROSS CROPS --------------
m1_sp = np.mean(data_1t['mh'])
m2_sp = np.mean(data_1t['ml'])
y_sp = np.mean(data_1t['y_agr'])

gain_y = 100*(y_sp-y_mom)/y_mom
print("Agricultural output increases by "+str(round(gain_y,2))+"%")

### increase intermediates -----------
m_sp = (m1_sp+m2_sp)
gain_m =100*( m_sp-(m1_mom+m2_mom))/(m1_mom+m2_mom)
print("Agricultural investment increases by "+str(round(gain_m,2))+"%")

outcomes = []
y_gain = [] 
for var in ['y_agr','yl','yh']:
    
    avg_gain = ( np.mean(data_1t[var]) - np.mean(bench_ss[var]))/np.mean(bench_ss[var]) 
    y_gain.append(avg_gain)
outcomes.append(y_gain)    

m_gain = []
for var in ['m','ml','mh']:   
    
    avg_gain = ( np.mean(data_1t[var]) - np.mean(bench_ss[var]))/np.mean(bench_ss[var]) 
    m_gain.append(avg_gain)
    
outcomes.append(m_gain)
column_names = ['Agg', 'Low Crops', 'High Crops']
index_names = ['y','m']
z_table = pd.DataFrame(outcomes, index=index_names,columns=column_names)    
z_table = z_table
z_table =  z_table*100
print('TABLE 8: AGRIC GAINS')
pd.options.display.float_format = '{:,.2f}'.format
print(z_table.to_latex(float_format="%.2f"))


# AGRICULTURAL GAINS ALONG PERMANENT COMPONENT ----------------

outcomes_z = []
for var in ['m','mh','ml','y_agr','yh','yl']:
    
    var_gain = [] 
    avg_gain = ( np.mean(data_1t[var]) - np.mean(bench_ss[var]))/np.mean(bench_ss[var]) 
    
    z_data_sp =  data_1t[['z',var]].groupby(by='z').mean()
    z_data_bmk =  bench_ss[['z',var]].groupby(by='z').mean()
    z_data_bmk.reset_index(inplace=True)
    z_data_sp.reset_index(inplace=True)
    z_gain =  (z_data_sp.iloc[:,1]-z_data_bmk.iloc[:,1])/z_data_bmk.iloc[:,1]
    blu = np.array(z_gain)
    var_gain.append(avg_gain)
    var_gain.extend(blu)
    
    outcomes_z.append(var_gain)

column_names = ['Avg','z1','z2','z3','z4','z5']
index_names = ['m','mh','ml','y','yh','yl']
z_table = pd.DataFrame(outcomes_z, index=index_names,columns=column_names)    
z_table = z_table.T
z_table =  z_table*100
print('TABLE 9: AGRIC GAINS ALONG PERMANENT COMPONENT')
print(z_table.to_latex(float_format="%.2f"))


#% CONSUMPTION AND WELFARE ==================================

## The aggregate effects of Completing Markets ------------------------------
aggreg_df = bench_ss[['y_agr','y','a','c']].mean().to_frame()
aggreg_df.columns = ['IM']
aggreg_df['CM'] = data_1t[['y_agr','a','y','c']].mean()
aggreg_df['diff'] = (aggreg_df['CM']-aggreg_df['IM'])
aggreg_df['%diff'] = aggreg_df['diff']/aggreg_df['IM']*100

print('  ')
print('TABLE 10: AGGREGATE GAINS')
print(aggreg_df.to_latex(float_format="%.2f"))


data_sim['lnc'] = np.log(data_sim['c'])

# Income and Consumption risk --------------
vol_ci = (data_sim.groupby(by='hh')[['lny','lnc']].var()).mean(axis=0)
vol_ci_sp = (data_sp.groupby(by='hh')[['lny','lnc']].var()).mean(axis=0)
vol_change = []

var_names = ['Income', 'Consumption']

for i, var in enumerate(var_names):
    share_risk = 100*(vol_ci_sp[i] - vol_ci[i])/vol_ci[i]
    vol_change.append(share_risk)
    
    

df_vol = pd.DataFrame({
    'Vars': var_names,
    'IM': vol_ci.values,  # Use .values to extract data from Series as a list
    'CM': vol_ci_sp.values,
    'Change': vol_change})





sum_eco = ((bench_ss[['y','a','c']]).iloc[1:,:]).describe()
sum_norisk = ((data_1t[['y','a','c']]).iloc[1:,:]).describe()
summary_sp = pd.concat([sum_norisk, sum_eco], axis=1)




## The Distributional effects of Completing Markets
outcomes = []
for var in ['y_agr','y','c']:


    perc_gains = []
    gini_gain =  (gini(data_1t[var]) - gini(bench_ss[var]))/gini(bench_ss[var])   
    perc_gains.append(gini_gain)
    percentiles = [1, 5, 10, 25, 50]
    

    for p in percentiles:
        data_p = bench_ss.loc[bench_ss[var] <= np.percentile(bench_ss[var].dropna(), p)]
        data_p2 = data_1t.loc[data_1t[var] <= np.percentile(data_1t[var].dropna(), p)]
        perc_gains.append((np.mean(data_p2[var]) - np.mean(data_p[var]))/np.mean(data_p[var]))
        
                  
        
    percentiles = [50, 75, 90, 95, 99]
    
    for p in percentiles:
        data_p = bench_ss.loc[ bench_ss[var] >= np.percentile(bench_ss[var].dropna(), p)]
        data_p2 = data_1t.loc[ data_1t[var] >= np.percentile(data_1t[var].dropna(), p)]
        perc_gains.append((np.mean(data_p2[var]) - np.mean(data_p[var]))/np.mean(data_p[var]))
        

    outcomes.append(perc_gains)
    
    
column_names = ['Gini', '1', '5', '10', '25', '50', '50', '25', '10', '5', '1']
index_names = ['Agric Outp', 'Income',  'Consumption']
data_gains = pd.DataFrame(outcomes, index=index_names,columns=column_names)

print('  ')
print('TABLE 11: CROSS-SECTIONAL GAINS')
print((100*data_gains).to_latex(float_format="%.2f"))



data_sim['V2'] = np.nan
data_sp['V2'] = np.nan

data_sim['t'] = data_sim['t']-110
data_sp['t'] = data_sp['t']-50

rho = cp.rho
#rho=1.01

# using saved value function the number is too high

# Let's recompute the value function:
for t in [0,1,2,3,4,5,6,7,8,9]:
    data_sim.loc[data_sim['t']==t,'V2'] = bbeta**(t)*cp.u_cont(data_sim.loc[data_sim['t']==t,'c'])
    data_sp.loc[data_sp['t']==t,'V2'] = bbeta**(t)*cp.u_cont(data_sp.loc[data_sp['t']==t,'c'])

data_sim.set_index(['hh','t'],inplace=True)
data_sp.set_index(['hh','t'],inplace=True)
w2_im = np.mean(data_sim.groupby('hh')['V2'].sum())
w2_cm = np.mean(data_sp.groupby('hh')['V2'].sum())

cev2 = (w2_cm/w2_im)**(1/(1-rho)) -1


print('The welfare gain in cev of completing markets is '+str(100*cev2))


data_sim.reset_index(inplace=True)
data_sp.reset_index(inplace=True)

#check welfare number
print('Check on the welfare number')
for t in [0,1,2,3,4,5,6,7,8,9]:
    print('Taking ',t,' periods')
    df1 = data_sim.loc[data_sim['t']<=t]
    df2 = data_sp.loc[data_sim['t']<=t]
    df1.set_index(['hh','t'],inplace=True)
    df2.set_index(['hh','t'],inplace=True)
    w2_im = np.mean(df1.groupby('hh')['V2'].sum())
    w2_cm = np.mean(df2.groupby('hh')['V2'].sum())
    cev2 = (w2_cm/w2_im)**(1/(1-rho)) -1
    print('The welfare gain in cev of completing markets is '+str(100*cev2))


