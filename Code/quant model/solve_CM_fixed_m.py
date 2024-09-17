# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:06:36 2021

@author: rodri
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_data.py').resolve().parent.parent # to run the file alone (not through master) remove 1 .parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)
my_dirct = dirct+'/quant model/'
os.chdir(my_dirct)


folder2 = dirct+'/Results/'
folder= dirct+'Results/SP/'


from model_class import HH_model


## Data Benchmark economy
data_sim =  pd.read_csv(folder2+'IM_solution_10periods.csv.gz')
#data_sim =  pd.read_csv('/home/rodri/Dropbox/JMP/figures/model/economy/states_stationarytemp.csv')


bench_ss = data_sim.loc[data_sim['t']==np.max(data_sim['t'])]
bench_ss['y_agr'] = bench_ss[['yh','yl']].sum(axis=1) 
bench_ss['m'] = bench_ss[['mh','ml']].sum(axis=1) 

outliers = bench_ss.loc[bench_ss['budget']>40,'hh']

bench_ss = bench_ss[~bench_ss['hh'].isin(outliers)]

# level of intermediates in the benchmark economy
m_bmk = np.mean(bench_ss['m'] )


#Import model class
### Import model parameters, grids, functions.
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
m_bar = m_bmk/p

mh_total = np.mean(mh_star)
ml_total = np.mean(ml_star)

mh_total2 = sum(mh_star)
ml_total2 = sum(ml_star)

m_total = mh_total+ml_total

mh_sh = mh_total/m_total
ml_sh = ml_total/m_total

mh_shares = mh_star/mh_total2
ml_shares = ml_star/ml_total2

mh_con = mh_shares*(mh_sh*m_bar)
ml_con = ml_shares*(ml_sh*m_bar)

a= z_grid*A*mh_con**alpha +z_grid*B*ml_con**gamma 
outp_fixed_m = p_z@a

b= z_grid*A*mh_star**alpha +z_grid*B*ml_star**gamma 
outp_cm = p_z@b

eff_gain = outp_fixed_m/outp_cm

print('===========================================================================')
print('The efficiency gains represent',100*eff_gain,'of the total productivity increase')
print('===========================================================================')

#Check works.
sum(mh_con)+sum(ml_con) - m_bar

# MP equalize?

mpl =  (gamma*B*z_grid*ml_con**(gamma-1))
mph = alpha*A*z_grid*mh_con**(alpha-1)

# marginal products equalize?
print(mpl)
print(mph)
print('Yes! MP equalize across agents and across technologies')


