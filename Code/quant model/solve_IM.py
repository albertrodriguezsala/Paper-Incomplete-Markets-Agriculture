# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:57:53 2020

@author: rodri
"""

# =============================================================================
 # SOLVING THE BENCHMARK MODEL: INCOMPLETE MARKETS
# =============================================================================
'''  
(1) SOLVE THE HOUSEHOLD PROBLEM AND PRODUCES FIGURE 3: HOUSEHOLD POLICY FUNCTIONS.
(2) SOLVE THE ECONOMY AND STORES SIMULATE DATA IN FILES
        (2.1) IM_solution_10periods.csv.gz   (stationary panel with 10 periods)
        (2.2) IM_solution_stationary.csv  (stationary cross-section)
(3) THIS IS THE FILE I USED FOR THE CALIBRATION BY CHANGING PARAMETER VALUES IN model_class.py.
    Current default values are the calibrated the ones in paper. Can be easily at calling the the model class: cp.HH_model() or in model_class default values
'''


import numpy as np
import os
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_quantmodel.py').resolve().parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)
    
my_dirct = dirct+'/quant model/'
os.chdir(my_dirct)
folder = dirct+'/Results/'

from quantecon import tic, toc 

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from matplotlib import pyplot as plt


from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import seaborn as sns


# My modules
from fixed_point import compute_fixed_point
from model_class import HH_model
from model_class import gini
from scipy import interpolate 


import matplotlib as mpl
fm = mpl.font_manager
colormap = plt.cm.Dark2.colors


mpl.rcParams['figure.figsize'] = (8.6, 6.4)
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
colormap = plt.cm.Dark2.colors

#%matplotlib inline

def my_lin(lb, ub, steps, spacing=2):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)])


### Set-up:
np.random.seed(23)
save = True
save_fig=True
plot = False
save_policies = False
save_simulation = False
plot_agg = False

### Import model parameters, grids, functions.
cp = HH_model(N_x=100, x_min=25, x_max=25000,N=1000000)  #N =1000000


### Accuracy and speed settings
num_cores =   int(0.4*mp.cpu_count())
T= cp.T
N= cp.N


x_min = cp.x_min
x_max = cp.x_max
# Initialize model =================================================================
# Parameters
A,B = cp.A, cp.B
alpha, gamma, rho, bbeta = cp.alpha, cp.gamma, cp.rho, cp.bbeta
r, p = cp.r, cp.p


SIGMA = cp.SIGMA
sig_theta = cp.sig_theta
sig_eps = cp.sig_eps
corr_shocks = cp.corr_shocks
sig_z = cp.sig_z


p_z = cp.p_z

#grids
N_x, N_a, N_m1, N_m2 = cp.N_x, cp.N_a, cp.N_m1, cp.N_m2
a_max,m1_max, m2_max = cp.a_max, cp.m1_max, cp.m2_max
b, m1_min, m2_min = cp.b, cp.m1_min, cp.m2_min
x_grid, a_grid, m1_grid, m2_grid = cp.x_grid, cp.a_grid, cp.m1_grid, cp.m2_grid


### Shocks Integration: 5-nodes Gauss-Hermite quadrature rule
SIGMA = cp.SIGMA

theta_j, eps_j, w_j = cp.theta_j, cp.eps_j, cp.w_j
# Krueger-Mitman-Perri use 3 nodes: I find it is to loow (0,98 not too close of 1)
# I use 5 that has expected value 0.99986

x, y = np.random.multivariate_normal([0,0], SIGMA, 500000).T
theta = np.exp(x-SIGMA[0,0]/2)
eps = np.exp(y-SIGMA[1,1]/2)


### permanent productivity -----
N_z, z_grid= cp.N_z, cp.z_grid
cp.p_z@cp.z_grid

### Non-agricultural stochastic income ------
pi_star = cp.pi_yna_star
N_yna, yna_grid, pi_yna = cp.N_yna, cp.yna_grid, cp.pi_yna



## Profit maximization
mh_star =  (p/(z_grid*A*bbeta*alpha))**(1/(alpha-1))
ml_star =  (p/(z_grid*B*bbeta*gamma))**(1/(gamma-1))

m = p*np.mean(mh_star)
ml = p*np.mean(ml_star)

# functions
u_cont = cp.u_cont
y1_func = cp.y1
y2_func = cp.y2


#%

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

 
# Set empty Value and Policy functions: ===
V_new = np.empty((N_x, N_z, N_yna))

## Initial Guess
V_guess = np.zeros((N_x,N_z, N_yna))
   
#V_guess = np.load(folder+'V_function100.npy')

### PARALLELIZATION  ==========================================================

# Define class for the model
class modelState(object):
    def __init__(self, N_x, N_z,N_yna, A, B,alpha, gamma, bbeta, p, r, w_j, pi_yna, x_grid,
                 z_grid, yna_grid,theta, eps,b):
       
       self.N_x = N_x
       self.N_z = N_z
       self.N_yna = N_yna
       self.A = A
       self.B = B
       self.alpha = alpha
       self.gamma = gamma
       self.bbeta = bbeta
       self.p = p
       self.r = r
       self.b = b
       self.pi_yna = pi_yna
       self.w_j = w_j
       self.theta = theta
       self.eps =  eps
       self.x_grid = x_grid
       self.z_grid = z_grid
       self.yna_grid = yna_grid
       


# Given a indexed state of today it computes the optimal value function V(index).
def V_i(V, index):
    i_x, i_z, i_yna = np.unravel_index(index, (N_x, N_z, N_yna)) 
    
    v_func0 = interpolate.CubicSpline(x_grid, V[:,i_z,0])
    v_func1 = interpolate.CubicSpline(x_grid, V[:,i_z,1])
    v_func2 = interpolate.CubicSpline(x_grid, V[:,i_z,2])
    v_func3 = interpolate.CubicSpline(x_grid, V[:,i_z,3])
    v_func4 = interpolate.CubicSpline(x_grid, V[:,i_z,4])
    
    def objective(X):
        a = X[1]
        c = X[0]
        m1 = X[2]
        m2 = X[3]
        
        if m1<0 or m2<0 or c<1e-3:
            return 1e+40
        
        else:
            return -(u_cont(c) +bbeta*(pi_yna[i_yna,0]*w_j@v_func0((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma) 
                           +pi_yna[i_yna,1]*w_j@v_func1((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)  
                            +pi_yna[i_yna,2]*w_j@v_func2((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)
                             +pi_yna[i_yna,3]*w_j@v_func3((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)
                            +pi_yna[i_yna,4]*w_j@v_func4((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)))                                                                                                                                                                                                                                

  
    
    bound_amin = b 
    bound_amax = x_grid[i_x]+yna_grid[i_yna]
    bound_cmin =  1
    bound_cmax = x_grid[i_x]+yna_grid[i_yna]
 
    ml_guess = 0.5*mh_star[i_z] 
    mh_guess = 0.5*ml_star[i_z] 
    c_guess = max(0.4*(x_grid[i_x]+yna_grid[i_yna]),25)
    a_guess = max((x_grid[i_x]+yna_grid[i_yna]) -c_guess -p*mh_guess -p*ml_guess,b+0.001)
 
    ml_low = 0.1
    mh_low = 0.1
    mh_max =1.25*mh_star[i_z]   
    ml_max= 1.25*ml_star[i_z]
    tol_error = 1e-13
    
    
    
    bnds = Bounds([bound_cmin, bound_amin, mh_low, ml_low], [bound_cmax,bound_amax, mh_max , ml_max])
    linear_constraint = LinearConstraint(A=[[1,1,p,p]], lb=(x_grid[i_x]+yna_grid[i_yna]), ub=(x_grid[i_x]+yna_grid[i_yna]))
    x0 = [c_guess, a_guess, mh_guess, ml_guess]
    res = minimize(objective, x0,  method='trust-constr', bounds=bnds, constraints=linear_constraint, jac='2-point', options={'factorization_method':'SVDFactorization','maxiter': 15000, 'verbose':0, 'gtol':tol_error, 'xtol':tol_error} )
   
    return -res.fun
 

### Parallelization: Computes V_i for each state on parallel across the CPUs. Then, groups together results to have V(x,z,y_na)    
def V_parallel(V):
    
    results = Parallel(n_jobs=num_cores, verbose=0)(delayed(V_i)(index=i, V=V) for i in range(0,N_x*N_z*N_yna))
    V_new = np.array(results)
    #mask = np.isnan(V_new)
    #V_new[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), V_new[~mask])
    V_new.shape = (N_x, N_z, N_yna) 

    return V_new


### Initialize states
states = modelState(N_x, N_z,N_yna, A, B,alpha, gamma, bbeta, p, r, w_j, pi_yna, x_grid,
                 z_grid, yna_grid,theta_j, eps_j,b)

A, B,alpha, gamma, bbeta, p, r = states.A, states.B, states.alpha, states.gamma,  states.bbeta, states.p, states.r
N_x, N_z, N_yna = states.N_x, states.N_z, states.N_yna 
x_grid, z_grid, theta_j, eps_j, yna_grid = states.x_grid, states.z_grid, states.theta, states.eps, states.yna_grid
w_j, pi_yna = states.w_j, states.pi_yna
b = states.b



### Iterate on the Bellman equation till convergence Value Function is achieved:
# 2.7E-04 WORKING WELL
#V= V_guess
V = compute_fixed_point(V_parallel, V_guess, max_iter=150, error_tol=9e-5, save=save, save_name= 'v_itera',folder=folder)

#150




#%  ==================== Get policies =========================================



def policies_i(V, index):
    i_x, i_z, i_yna = np.unravel_index(index, (N_x, N_z, N_yna)) 
    
    v_func0 = interpolate.CubicSpline(x_grid, V[:,i_z,0])
    v_func1 = interpolate.CubicSpline(x_grid, V[:,i_z,1])
    v_func2 = interpolate.CubicSpline(x_grid, V[:,i_z,2])
    v_func3 = interpolate.CubicSpline(x_grid, V[:,i_z,3])
    v_func4 = interpolate.CubicSpline(x_grid, V[:,i_z,4])
    
    def objective(X):
        a = X[3]
        c = X[2]
        m1 = X[0]
        m2 = X[1]
        
        if m1<0 or m2<0 or c<1e-1:
            return 1e+40
        
        else:
            return -(u_cont(c) +bbeta*(pi_yna[i_yna,0]*w_j@v_func0((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma) 
                           +pi_yna[i_yna,1]*w_j@v_func1((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)  
                            +pi_yna[i_yna,2]*w_j@v_func2((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)
                             +pi_yna[i_yna,3]*w_j@v_func3((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)
                            +pi_yna[i_yna,4]*w_j@v_func4((1+r)*a +theta_j*z_grid[i_z]*A*m1**alpha +eps_j*z_grid[i_z]*B*m2**gamma)))                                                                                                                                                                                                                                    

  

    bound_amin = b 
    bound_amax = x_grid[i_x]+yna_grid[i_yna]
    bound_cmin =  1
    bound_cmax = x_grid[i_x]+yna_grid[i_yna]
 
    ml_guess = 0.5*mh_star[i_z] 
    mh_guess = 0.5*ml_star[i_z] 
    c_guess = max(0.4*(x_grid[i_x]+yna_grid[i_yna]),25)
    a_guess = max((x_grid[i_x]+yna_grid[i_yna]) -c_guess -p*mh_guess -p*ml_guess,b+0.001)
 
    ml_low = 0.1
    mh_low = 0.1
    mh_max =5*mh_star[i_z]   
    ml_max= 5*ml_star[i_z]
    tol_error = 1e-13
 
   
      
    bnds = Bounds([mh_low, ml_low,bound_cmin, bound_amin], [mh_max, ml_max,bound_cmax,bound_amax])
    linear_constraint = LinearConstraint(A=[[p,p,1,1]], lb=(x_grid[i_x]+yna_grid[i_yna]), ub=(x_grid[i_x]+yna_grid[i_yna]))
    x0 = [mh_guess, ml_guess,c_guess, a_guess]
    res = minimize(objective, x0,  method='trust-constr', bounds=bnds, constraints=linear_constraint, jac='2-point', options={'factorization_method':'SVDFactorization','maxiter': 30050, 'verbose':0, 'gtol':tol_error, 'xtol':tol_error} )
    
    
    ga = res.x[3]
    gc = res.x[2]
    gmh = res.x[0]
    gml =  res.x[1]
    vnew = -res.fun
    
    
    return ga,  gmh, gml, gc, vnew



def policies(V):
    res = Parallel(n_jobs=num_cores, verbose=1)(delayed(policies_i)(index=i,  V=V) for i in range(0,N_x*N_z*N_yna))
    policy_a = np.array([item[0] for item in res])
    policy_m1 = np.array([item[1] for item in res])
    policy_m2= np.array([item[2] for item in res])
    policy_c= np.array([item[3] for item in res])
    V_new = np.array([item[4] for item in res])
    policy_a.shape = (N_x,N_z,N_yna)
    policy_m1.shape = (N_x,N_z,N_yna)
    policy_m2.shape = (N_x,N_z,N_yna)
    policy_c.shape = (N_x,N_z,N_yna)
    V_new.shape = (N_x,N_z,N_yna)
                                
    return policy_a, policy_m1, policy_m2, policy_c, V_new


#### Compute policy functions
tic()
g_a, g_m1, g_m2, g_c, V_next = policies(V)
toc()


def g_a_function(x,i_z,i_yna):
    return np.interp(x,x_grid, g_a[:,i_z,i_yna])

def g_m1_function(x,i_z,i_yna):
    return np.interp(x,x_grid, g_m1[:,i_z,i_yna])

def g_m2_function(x,i_z,i_yna):
    return np.interp(x,x_grid, g_m2[:,i_z,i_yna])

def g_c_function(x,i_z,i_yna):
    return np.interp(x,x_grid, g_c[:,i_z,i_yna])


def g_v_function(x,i_z,i_yna):
    return np.interp(x,x_grid, V_next[:,i_z,i_yna])

    

    
#-----------------------    


z_types = ['L','L-M','M-L','M-H','H']

def plot_policy_2d(grid, policy,  policy_name,save_name, ylim=False, legend=False, save=False, line_45=False, folder=folder):
        
        fig,ax = plt.subplots()
        color=iter(plt.cm.rainbow(np.linspace(0,1,N_z*N_yna)))
        for i_z in range(0,N_z):
            for i_yna in range(0,N_yna):
                c=next(color)
                ax.plot(grid, policy[:,i_z,i_yna],color=c, label='yna='+str(i_yna)+', z='+z_types[i_z])
        if line_45 == True:
            ax.plot(grid,grid, linestyle='dashed', label='45 line')
        ax.set_xlabel('Cash on hand (X)')
        ax.set_ylabel(policy_name)
        ax.set_xlim((min(x_grid),15000))
        if ylim == True:
            ax.set_ylim((-500,17000))
        if legend==True:
            ax.legend(fontsize=15, ncol=2)
        if save==True:
            fig.savefig(folder+'figures/model/economy/'+save_name+'.png')  
        plt.show() 


def plot_policy_2d_inputs(grid, policy, policy_name,save_name, m_star, save=False, line_45=False, folder=folder):
        
        fig,ax = plt.subplots()
        color=iter(plt.cm.rainbow(np.linspace(0,1,N_z*N_yna)))
        for i_z in range(0,N_z):
            for i_yna in range(0,N_yna):
                c=next(color)
                ax.plot(grid, policy[:,i_z,i_yna], color=c, label='yna='+str(i_yna)+', z='+z_types[i_z])
            ax.plot(grid,m_star[i_z]*np.ones(len(grid)), color=c, linestyle='dashed', label='Profit Max: z='+z_types[i_z])
        ax.set_xlabel('Cash on hand (X)')
        ax.set_ylim((0,1.5*m_star[N_z-1]))
        ax.set_xlim((min(x_grid),15000))
        ax.set_ylabel(policy_name)
        
        if save==True:
            fig.savefig(folder+'figures/model/economy/'+save_name+'.png')  
        plt.show() 
        
        
plot_policy_2d(grid=x_grid, policy=V_next, save=save_fig, legend=True, policy_name='$V(x,z,y_{na})$', save_name='Value_function')
plot_policy_2d(grid=x_grid,policy=g_a, save=save_fig, ylim=True, line_45=True, policy_name="$a(x,z,y_{na})$", save_name='Assets_Policy') 
plot_policy_2d(grid=x_grid,policy=g_c, save=save_fig,policy_name="$c(x,z,y_{na})$",  save_name='Consumption_Policy')
plot_policy_2d_inputs(grid=x_grid,policy=g_m1, m_star=mh_star,policy_name="$m_h'(x,z,y_{na})$", save=save_fig, save_name='Input_High_Policy')
plot_policy_2d_inputs(grid=x_grid, policy=g_m2, m_star=ml_star,policy_name="$m_{\ell}'(x,z,y_{na})$",  save=save_fig,  save_name='Input_Low_Policy') 


def plot_wedge(grid, g_m1, g_m2, save=False, folder=folder):
        
    fig,ax = plt.subplots()
    color=iter(plt.cm.rainbow(np.linspace(0,1,N_z*N_yna)))
    for i_z in range(0,N_z):
        for i_yna in range(0,N_yna):
            c=next(color)
            MP_h = z_grid[i_z]*A*alpha*g_m1[:,i_z,i_yna]**(alpha-1)
            MP_l =z_grid[i_z]*B*gamma*g_m2[:,i_z,i_yna]**(gamma-1)            
            ax.plot(grid, MP_h/MP_l, color=c, label='yna='+str(i_yna)+', z='+z_types[i_z])
            ax.set_xlabel('Cash on hand (X)')
    #ax.set_title('Marginal Product Wedge')
    #ax.legend(loc='lower right', fontsize='x-small')
    ax.set_ylim((0.95,1.6))
    ax.set_xlim((min(x_grid),15000))
    if save==True:
        fig.savefig(folder+'figures/model/economy/MP_wedge_policies.png')  
    plt.show() 



plot_wedge(grid=x_grid,g_m1=g_m1,g_m2=g_m2)


#%%   

if save==True:
    np.save(folder+'V_function'+str(N_x), V_next)


    



#%% Compute stationary distribution by Montecarlo simulation (Parallelalized)


def next_yna(current_yna):    
    return np.random.choice(yna_grid, p=pi_yna[current_yna, :])


class model_simulation(object):
    def __init__(self, N, T,N_z, yna_grid, z_grid, SIGMA, theta, eps,w_j, policies, r, functions):
		 
       self.N = N
       self.T = T
       self.yna_grid = yna_grid
       self.z_grid = z_grid
       self.iz_state = np.random.choice(a=N_z,size=N,p=p_z)
       self.SIGMA = SIGMA
       self.r = r
       self.mh_star = mh_star
       self.ml_star = ml_star
       self.sig_theta = SIGMA[0,0]
       self.sig_eps = SIGMA[1,1]
       self.theta = theta
       self.eps = eps
       self.w_j = w_j
       self.f_ga = policies[0]
       self.f_gm1 = policies[1]
       self.f_gm2 = policies[2]
       self.f_gc = policies[3]
       self.f_v = policies[4]
       self.y1_func = functions[0]
       self.y2_func = functions[1]
       self.next_yna = functions[2]
       


policies = [g_a_function,g_m1_function,g_m2_function,g_c_function,g_v_function]
#policies = [GA_function,GM1_function,GM2_function,GC_function]
functions = [y1_func, y2_func, next_yna]

sim = model_simulation( N, T,N_z, yna_grid, z_grid, SIGMA, theta_j, eps_j,w_j,policies, r, functions)
N = sim.N
T = sim.T
r = sim.r
SIGMA = sim.SIGMA
sig_theta = sim.sig_theta
sig_eps = sim.sig_eps
theta_j = sim.theta
eps_j = sim.eps
w_j = sim.w_j
f_ga = sim.f_ga
f_gm1 = sim.f_gm1
f_gm2 = sim.f_gm2
f_gc = sim.f_gc
f_v =sim.f_v
y1_func = sim.y1_func
y2_func = sim.y2_func
next_yna = sim.next_yna
z_grid = sim.z_grid
iz_state = sim.iz_state



#### Parall
def sim_outcomes_n(index):
    x_state = np.empty(T)
    yna_state = np.empty(T)
    z_state = np.empty(T)
    budget_state = np.empty(T)
    a_state, m1_state, m2_state, c_state = np.empty(T),np.empty(T),np.empty(T),np.empty(T)
    yh_state, yl_state, y_state, time = np.empty(T),np.empty(T),np.empty(T),np.empty(T)
    yna_list = yna_grid.tolist()
    theta_state, eps_state = np.empty(T), np.empty(T)
    v_state = np.empty(T)
    
    hh = index*np.ones((T,1))
    
    #permanent productivity
    iz = iz_state[index]
    z_state[:] = z_grid[iz]
    
    #shocks
    j_shock = np.random.choice(a=len(theta_j),size=1,p=w_j)
    theta = theta_j[j_shock]
    eps = eps_j[j_shock]
    yna_state[0] = yna_grid[1]
    x_state[0] = (1+r)*1300 +y1_func(z_grid[iz],mh_star[iz],theta) +y2_func(z_grid[iz],ml_star[iz],eps) 
    # using a initial wealth close to cvalibration target. Note that eith higher values the simulation was breaking down
   
    
    for t in range(1, T):
        j_shock = np.random.choice(a=len(theta_j),size=1,p=w_j)
        theta = theta_j[j_shock]
        eps = eps_j[j_shock]
        
        theta_state[t] = theta
        eps_state[t] = eps
        
        yna_it = int(yna_list.index(yna_state[t-1]))
        yna_state[t] = next_yna(yna_it)
        m1_state[t] = f_gm1(x_state[t-1],iz,yna_it)
        m2_state[t] = f_gm2(x_state[t-1],iz,yna_it)
        a_state[t] = f_ga(x_state[t-1],iz,yna_it)
        v_state[t] = f_v(x_state[t-1],iz,yna_it)
        theta_state[t] = theta
        eps_state[t] = eps
        
        yh_state[t] = y1_func(z_grid[iz],m1_state[t],theta)
        yl_state[t] = y2_func(z_grid[iz],m2_state[t], eps) 
        y_state[t] = yh_state[t] +yl_state[t] +yna_state[t] 
        x_state[t] = (1+r)*a_state[t] +yh_state[t] +yl_state[t] 
        c_state[t] = f_gc(x_state[t],iz,int(yna_list.index(yna_state[t])))
        budget_state[t] = np.abs(x_state[t-1] + yna_state[t-1] -a_state[t] -c_state[t-1] -p*m1_state[t] -p*m2_state[t])
        time[t] = t

     
    return hh, time, yh_state, yl_state, y_state, a_state, m1_state, m2_state, c_state, x_state, yna_state, z_state, budget_state, theta_state, eps_state, v_state




def sim_outcomes_parallel(N,T):
    
    res = Parallel(n_jobs=num_cores, verbose=0)(delayed(sim_outcomes_n)(index=i) for i in range(0,N))
    
    hh = np.array([item[0] for item in res])
    time = np.array([item[1] for item in res])
    yh_state = np.array([item[2] for item in res])
    yl_state = np.array([item[3] for item in res])
    y_state = np.array([item[4] for item in res])
    a_state = np.array([item[5] for item in res])
    m1_state = np.array([item[6] for item in res])
    m2_state = np.array([item[7] for item in res])
    c_state = np.array([item[8] for item in res])
    x_state = np.array([item[9] for item in res])
    yna_state = np.array([item[10] for item in res])
    z_state = np.array([item[11] for item in res])
    budget_state = np.array([item[12] for item in res])
    theta_state = np.array([item[13] for item in res])
    eps_state = np.array([item[14] for item in res])
    v_state = np.array([item[15] for item in res])
    
    hh.shape, time.shape, yh_state.shape, yl_state.shape, y_state.shape,   = (N,T),(N,T),(N,T),(N,T),(N,T)
    a_state.shape, m1_state.shape, m2_state.shape, c_state.shape, x_state.shape, yna_state.shape  = (N,T),(N,T),(N,T),(N,T),(N,T),(N,T)
    z_state.shape, budget_state.shape = (N,T), (N,T)
    theta_state.shape, eps_state.shape, v_state.shape = (N,T), (N,T), (N,T)
    
    hh, time, yh_state, yl_state, y_state = hh.T, time.T, yh_state.T, yl_state.T, y_state.T
    a_state, m1_state, m2_state, c_state, x_state, yna_state  = a_state.T, m1_state.T, m2_state.T, c_state.T, x_state.T, yna_state.T
    z_state, budget_state = z_state.T, budget_state.T
    theta_state, eps_state, v_state = theta_state.T, eps_state.T, v_state.T
    t_min = 20 # min(200,T-2)
    data_simulation_dict = [('hh', (hh[T-t_min:T,:].transpose()).flatten()),
                        ('t', (time[T-t_min:T,:].transpose()).flatten()),
                        ('y', (y_state[T-t_min:T,:].transpose()).flatten()),
                        ('yh', (yh_state[T-t_min:T,:].transpose()).flatten()),
                        ('yl', (yl_state[T-t_min:T,:].transpose()).flatten()),
                        ('X', (x_state[T-t_min:T,:].transpose()).flatten()),
                        ('yna', (yna_state[T-t_min:T,:].transpose()).flatten()),
                        ('z', (z_state[T-t_min:T,:].transpose()).flatten()), 
                        ('mh', p*(m1_state[T-t_min:T,:].transpose()).flatten()),
                        ('ml', p*(m2_state[T-t_min:T,:].transpose()).flatten()),
                        ('a', (a_state[T-t_min:T,:].transpose()).flatten()),
                        ('c', (c_state[T-t_min:T,:].transpose()).flatten()),
                        ('theta', (theta_state[T-t_min:T,:].transpose()).flatten()),
                        ('eps', (eps_state[T-t_min:T,:].transpose()).flatten()),
                        ('budget', (budget_state[T-t_min:T,:].transpose()).flatten()),
                        ('V', (v_state[T-t_min:T,:].transpose()).flatten()),]

            
    return pd.DataFrame.from_dict(dict(data_simulation_dict))

            
print('Simulation time:')
tic()
data_sim = sim_outcomes_parallel(N,T)
toc()

pd.value_counts(data_sim['t'])

data_sim = data_sim[data_sim['t']>T-11]

#print(data_sim['budget'].describe(percentiles=[0.1,0.2,0.5,0.8,0.9,0.95,0.99]))

outliers = data_sim.loc[data_sim['budget']>250,'hh']
data_sim = data_sim[~data_sim['hh'].isin(outliers)]


data_sim['t'].describe()

if save_simulation == True:
    data_sim.to_csv(folder+'IM_solution_10periods.csv.gz',compression='gzip')
    print('SIMULATION SAVED; states_stationary.csv')



#### Plot state variable distrib_ution last periods
fig, ax = plt.subplots(figsize=(8,6))
for t in range(1,7):
    x_data = data_sim.loc[data_sim['t']==T-t,'X']   
    sns.distplot(np.log(x_data+1), label='Period '+str(T-t))
#plt.title('Invariant distribution')
plt.xlabel('Cash-on-hand')
plt.ylabel("Density") 
fig.savefig(folder+'figures/model/economy/cvg_stationarity.png')  
plt.legend()
plt.show() 


mean_a = data_sim[['a','t']].groupby(by='t').mean()

fig, ax = plt.subplots()
ax.plot(range(1,len(mean_a)+1), mean_a, label='Wealth')
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Average across time')                  
plt.show()  


# =============================================================================
#  Economy outcomes
# =============================================================================

data_1t = data_sim.loc[data_sim['t']==T-1,]

if save==True:
    data_1t.to_csv(folder+'IM_solution_stationary.csv')

#print('Check budget constraint:')
#print(data_1t['budget'].describe(percentiles=[0.1,0.2,0.5,0.8,0.9,0.95,0.99]))

outliers = data_1t.loc[data_1t['budget']>40,'hh']
data_1t = data_1t[~data_1t['hh'].isin(outliers)]

#%%
### States distribution in stationarity  =============
data_1t['y_agr'] = data_1t[['yh','yl']].sum(axis=1) 
share_yh = np.mean(data_1t['yh']/data_1t['y_agr'])
share_yhl = np.mean(data_1t['yh'])/np.mean(data_1t['yl'])

check = pd.value_counts(data_1t['yna'])/len(data_1t)
### To logs
data_sim['lny_h'] = np.log(data_sim['yh'])
data_sim['lny_l'] = np.log(data_sim['yl'])
data_sim['lny'] = np.log(data_sim['y'])
data_sim['lnc'] = np.log(data_sim['c'])
data_sim['lna'] = np.log(data_sim['a'])

#TARGETED MOMENTS ===============================

# Averages
means = data_1t[['yh','yl','y','a']].mean()

# Risk
timevar = data_sim.groupby(by='hh')[['lny_h', 'lny_l']].cov()
timevar.reset_index(inplace=True)
blu = timevar.groupby(by='level_1').mean()
sd_yh = np.sqrt(blu.iloc[0,1])
sd_yl = np.sqrt(blu.iloc[1,2])

cov_yhyl = blu.iloc[1,1]/(sd_yh*sd_yl)
sigmas = [ blu.iloc[0,1], blu.iloc[1,2], cov_yhyl]

# Ineq
gini_ya = gini(data_1t['y_agr'])
momt_labels = ['yh', 'yl','y','a','vol_yh','vol_yl','corr_yhyl', 'gini_agric']

momt = []

momt.extend(means)
momt.extend(sigmas)
momt.append(gini_ya)

# Moments
momt_data = [ 651.12, 353.39,  1506, 1858, 1.05, 0.722, 0.115, 0.62]

df_momt = pd.DataFrame({'Moment':pd.Series(momt_labels),'Model':pd.Series(momt),'Data':pd.Series(momt_data)})
print(' CALIBRATION: MODEL VS DATA MOMENTS         ')
print('             ')
print(df_momt)





