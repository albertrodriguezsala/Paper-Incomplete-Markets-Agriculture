
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantecon import compute_fixed_point
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(os.getcwd())

dirct  = Path('Master_data.py').resolve().parent.parent # to run the file alone (not through master) remove 1 .parent


if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_dirct = dirct+'/quant model/'
os.chdir(my_dirct)
print(dirct) 
folder = dirct+'/Results/'
#folder = os.path.join(dirct, 'Results/')
  
from model_class import HH_model
import quantecon as qe
import seaborn as sns

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.optimize import minimize_scalar

from scipy import interpolate 

np.random.seed(23)

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
plt.rcParams['figure.subplot.left'] = 0.18
colormap = plt.cm.Dark2.colors
save=True
sys.path.append(dirct) 

### Set-up things ----------------------

#Import model class
### Import model parameters, grids, functions.
cp = HH_model(N_a=100, x_max=25000, N=1000000) #previous N=800000

# Parellelize usuing 70% of available cores
num_cores = int(0.4*mp.cpu_count())

### Accuracy 
tol_error = cp.tol_error

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
N = cp.N
p_z = cp.p_z

#grids
N_x, N_a, N_m1, N_m2 = cp.N_x, cp.N_a, cp.N_m1, cp.N_m2
a_max,m1_max, m2_max = cp.a_max, cp.m1_max, cp.m2_max
b, m1_min, m2_min = cp.b, cp.m1_min, cp.m2_min
x_grid, a_grid, m1_grid, m2_grid = cp.x_grid, cp.a_grid, cp.m1_grid, cp.m2_grid

a_grid = np.linspace(b, a_max, N_a)

### Shocks Integration: 5-nodes Gauss-Hermite quadrature rule
SIGMA = cp.SIGMA
theta_j, eps_j, w_j = cp.theta_j, cp.eps_j, cp.w_j
N=cp.N

y1 = np.vectorize(cp.y1)
y2 = np.vectorize(cp.y2)

### permanent productivity -----
N_z, z_grid= cp.N_z, cp.z_grid

### Non-agricultural stochastic income ------
pi_star = cp.pi_yna_star
N_yna, yna_grid, pi_yna = cp.N_yna, cp.yna_grid, cp.pi_yna
mean_yna = pi_star@yna_grid



# functions
u_cont = cp.u_cont
y1_func = cp.y1
y2_func = cp.y2


## Social planner solution:
mh_star = (p/(bbeta*z_grid*A*alpha))**(1/(alpha-1))
ml_star =  (p/(bbeta*z_grid*B*gamma))**(1/(gamma-1))
c_star =  z_grid*A*mh_star**alpha +z_grid*B*ml_star**gamma -p*mh_star -p*ml_star 



# Set empty Value and Policy functions: ===
V_next = np.empty((N_a, N_z, N_yna))
c_policy = np.empty((N_a, N_z, N_yna))
a_policy = np.empty((N_a, N_z, N_yna))

## Initial Guess
V_guess = np.zeros((N_a,N_z, N_yna))
#V_guess = np.load(folder+'V_sp.npy')


#Bellman equation
def bellman_operator(V,return_policies=False):
      
    
    for iz in range(0,N_z):
        v_func0 = interpolate.CubicSpline(a_grid, V[:,iz,0])
        v_func1 = interpolate.CubicSpline(a_grid, V[:,iz,1])
        v_func2 = interpolate.CubicSpline(a_grid, V[:,iz,2])
        v_func3 = interpolate.CubicSpline(a_grid, V[:,iz,3])
        for ia in range(0,N_a):
            for iyna in range(0,N_yna):
                #print(str(iz)+', '+str(ia)+", "+str(iyna))
                def criterion_func(X):  
                    
                    c_today = yna_grid[iyna] +(1+r)*a_grid[ia] -X
                    
                    if c_today<0:
                        return 1e+80

                    return  -(u_cont(c_star[iz] +c_today) +bbeta*(pi_yna[iyna,0]*v_func0(X) 
                           +pi_yna[iyna,1]*v_func1(X) 
                            +pi_yna[iyna,2]*v_func2(X) 
                             +pi_yna[iyna,3]*v_func3(X)))  
        
                a_max = yna_grid[iyna] +(1+r)*a_grid[ia] 
               
                result = minimize_scalar(criterion_func, bounds=(0, a_max), method='bounded')
                a_policy[ia,iz,iyna] =  result.x 
                c_policy[ia,iz,iyna] = c_star[iz] +yna_grid[iyna] +(1+r)*a_grid[ia] -a_policy[ia,iz,iyna]
                V_next[ia,iz,iyna] =  -result.fun   
                
    if return_policies==True:
        return V_next, a_policy, c_policy
    else:
        return V_next


V_star = compute_fixed_point(bellman_operator, V_guess, max_iter=150, error_tol=9e-5)
V, ga, gc = bellman_operator(V_star, return_policies=True)


  
def g_a_function(x,i_z,i_yna):
    return np.interp(x,a_grid, ga[:,i_z,i_yna])


def g_c_function(x,i_z,i_yna):
    return np.interp(x,a_grid, gc[:,i_z,i_yna])


def g_v_function(x,i_z,i_yna):
    return np.interp(x,a_grid, V[:,i_z,i_yna])


#-----------------------    


z_types = ['L','L-M','M', 'M-H','H']

def plot_policy_2d(grid, policy,  policy_name,save_name, save=False, legend=False, line_45=False, folder=folder):
        
        fig,ax = plt.subplots(figsize=(8.6,7))
        color=iter(plt.cm.rainbow(np.linspace(0,1,N_z*N_yna)))
        for i_z in range(0,N_z):
            for i_yna in range(0,N_yna):
                c=next(color)
                ax.plot(grid, policy[:,i_z,i_yna],color=c, label='yna='+str(i_yna)+', z='+z_types[i_z])
        if line_45 == True:
            ax.plot(grid,grid, linestyle='dashed', label='45 line')
        ax.set_xlabel('Assets (a)')
        ax.set_xlim((1e-1,10000))
        ax.set_ylabel(policy_name)
        #fig.subplots_adjust(left=0.15) 
        if legend==True:
            ax.legend(fontsize=15, ncol=2)
        #ax.legend(loc='lower right', fontsize='x-small')
        if save==True:
            fig.savefig(folder+save_name+'.png')  
        plt.show() 



plot_policy_2d(grid=a_grid, policy=V, save=False, legend=True, policy_name=' $V(a,z,y^{na})$ ', save_name='Vf_CM')
plot_policy_2d(grid=a_grid,policy=ga, save=False, line_45=True,policy_name=" $a'(a,z,y^{na})$ ", save_name='ga_CM') 
plot_policy_2d(grid=a_grid,policy=gc, save=False, policy_name=" $c(a,z,y^{na})$ ",  save_name='gc_CM')
 
#%%


## Data Benchmark model
data_sim =  pd.read_csv(folder+'IM_solution_10periods.csv.gz')
bench_ss = data_sim.loc[data_sim['t']==np.max(data_sim['t']),]
bench_ss['y_agr'] = bench_ss[['yh','yl']].sum(axis=1) 
bench_ss['m'] = bench_ss[['mh','ml']].sum(axis=1) 



N= len(bench_ss)
T=60

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def next_yna(current_yna):    
    return np.random.choice(yna_grid, p=pi_yna[current_yna, :])



class model_simulation(object):
    def __init__(self, N, T,N_z, yna_grid,a_grid, z_grid,SIGMA,theta,eps,w_j,policies,r,functions):
		 
       self.N = N
       self.T = T
       self.yna_grid = yna_grid
       self.a_grid = a_grid
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
       self.f_gm1 = policies[0]
       self.f_gm2 = policies[1]
       self.f_ga = policies[2]
       self.f_gc = policies[3]
       self.f_v = policies[4]
       self.y1_func = functions[0]
       self.y2_func = functions[1]
       self.next_yna = functions[2]


policies = [mh_star,ml_star,g_a_function, g_c_function, g_v_function]
functions = [y1_func, y2_func, next_yna]

sim = model_simulation(N,T,N_z,yna_grid,a_grid,z_grid,SIGMA,theta_j,eps_j,w_j,policies,r,functions)
N = sim.N
T = sim.T
r = sim.r
SIGMA = sim.SIGMA
sig_theta = sim.sig_theta
sig_eps = sim.sig_eps
theta_j = sim.theta
eps_j = sim.eps
w_j = sim.w_j
f_gm1 = sim.f_gm1
f_gm2 = sim.f_gm2
f_ga = sim.f_ga
f_gc = sim.f_gc
f_v = sim.f_v
y1_func = sim.y1_func
y2_func = sim.y2_func
next_yna = sim.next_yna
z_grid = sim.z_grid
iz_state = sim.iz_state




#### Parallelalize simulation----------------
def sim_outcomes_n(index):
    a_state = np.empty(T)
    yna_state = np.empty(T)
    z_state = np.empty(T)
    m1_state, m2_state, c_state = np.empty(T),np.empty(T),np.empty(T)
    yh_state, yl_state, y_state, time = np.empty(T),np.empty(T),np.empty(T),np.empty(T)
    yna_list = yna_grid.tolist()
    hh = index*np.ones((T,1))
    theta_state = np.empty(T)
    eps_state = np.empty(T)
    v_state = np.empty(T)
    #permanent productivity
    
    iz = iz_state[index]
    z_state[:] = z_grid[iz]
    
    #shocks
    j_shock = np.random.choice(a=len(theta_j),size=1,p=w_j)
    theta = theta_j[j_shock]
    eps = eps_j[j_shock]
    yna_state[0] = yna_grid[1]
    a_state[0] =800
    eps_state[0] = 1
    theta_state[0] = 1
    
    
    for t in range(1, T):
        j_shock = np.random.choice(a=len(theta_j),size=1,p=w_j).item()
        theta =  theta_j[j_shock]
        eps = eps_j[j_shock]
        
        yna_it = int(yna_list.index(yna_state[t-1].item()))
        yna_state[t] = next_yna(yna_it)
        m1_state[t] = mh_star[iz]
        m2_state[t] = ml_star[iz]
        a_state[t] = f_ga(a_state[t-1],iz,yna_it)
        v_state[t] = f_v(a_state[t-1],iz,yna_it)
        
        yh_state[t] = theta*z_grid[iz]*A*m1_state[t]**alpha
        yl_state[t] = eps*z_grid[iz]*B*m2_state[t]**gamma
        theta_state[t] = theta
        eps_state[t] = eps
        y_state[t] = yh_state[t] +yl_state[t] +yna_state[t]
       
        c_state[t] =  f_gc(a_state[t-1],iz,yna_it)
        time[t] = t
     
    return hh, time, yh_state, yl_state, y_state, m1_state, m2_state, c_state, yna_state, z_state, theta_state, eps_state, a_state, v_state




def sim_outcomes_parallel(N,T):
    res = Parallel(n_jobs=num_cores, verbose=0)(delayed(sim_outcomes_n)(index=i) for i in range(0,N))
    
    hh = np.array([item[0] for item in res])
    time = np.array([item[1] for item in res])
    yh_state = np.array([item[2] for item in res])
    yl_state = np.array([item[3] for item in res])
    y_state = np.array([item[4] for item in res])
    m1_state = np.array([item[5] for item in res])
    m2_state = np.array([item[6] for item in res])
    c_state = np.array([item[7] for item in res])
    yna_state = np.array([item[8] for item in res])
    z_state = np.array([item[9] for item in res])
    theta_state = np.array([item[10] for item in res])
    eps_state = np.array([item[11] for item in res])
    a_state = np.array([item[12] for item in res])
    v_state = np.array([item[13] for item in res])
 
    hh.shape, time.shape, yh_state.shape, yl_state.shape, y_state.shape,   = (N,T),(N,T),(N,T),(N,T),(N,T)
    m1_state.shape, m2_state.shape, c_state.shape,  yna_state.shape, a_state.shape  = (N,T),(N,T),(N,T),(N,T),(N,T)
    z_state.shape= (N,T)
    theta_state.shape, eps_state.shape, v_state.shape = (N,T), (N,T), (N,T)
    
    hh, time, yh_state, yl_state, y_state,  = hh.T, time.T, yh_state.T, yl_state.T, y_state.T
    m1_state, m2_state, c_state, yna_state  =  m1_state.T, m2_state.T, c_state.T, yna_state.T
    z_state, v_state = z_state.T, v_state.T
    theta_state, eps_state, a_state = theta_state.T, eps_state.T, a_state.T
    t_min = 10
    data_simulation_dict = [('hh', (hh[T-t_min:T,:].transpose()).flatten()),
                        ('t', (time[T-t_min:T,:].transpose()).flatten()),
                        ('y', (y_state[T-t_min:T,:].transpose()).flatten()),
                        ('yh', (yh_state[T-t_min:T,:].transpose()).flatten()),
                        ('yl', (yl_state[T-t_min:T,:].transpose()).flatten()),
                        ('yna', (yna_state[T-t_min:T,:].transpose()).flatten()),
                        ('z', (z_state[T-t_min:T,:].transpose()).flatten()), 
                        ('mh', p*(m1_state[T-t_min:T,:].transpose()).flatten()),
                        ('ml', p*(m2_state[T-t_min:T,:].transpose()).flatten()),
                        ('theta', (theta_state[T-t_min:T,:].transpose()).flatten()),
                        ('eps', (eps_state[T-t_min:T,:].transpose()).flatten()),
                        ('a', (a_state[T-t_min:T,:].transpose()).flatten()),
                        ('V', (v_state[T-t_min:T,:].transpose()).flatten()),
                        ('c', (c_state[T-t_min:T,:].transpose()).flatten()),]

            
    return pd.DataFrame.from_dict(dict(data_simulation_dict))



print('Simulation time:')
qe.tic()
data_sp  = sim_outcomes_parallel(N,T)
qe.toc()


data_sp['y_agr'] = data_sp[['yh','yl']].sum(axis=1) 
data_sp['m'] = data_sp[['mh','ml']].sum(axis=1) 


data_1t = data_sp.loc[data_sp['t']==T-1,]

share_yh = np.mean(data_1t['yh']/data_1t['y_agr'])
share_yhl = np.mean(data_1t['yh'])/np.mean(data_1t['yl'])

### from income:
data_sp['lny_h'] = np.log(data_sp['yh'])
data_sp['lny_l'] = np.log(data_sp['yl'])
data_sp['lny'] = np.log(data_sp['y'])
data_sp['lnya'] = np.log(data_sp['y_agr'])
data_sp['lnc'] = np.log(data_sp['c']+np.abs(np.min(data_sp['c'])))


data_sp[['y_agr','yh','yl']].mean()

timevar = data_sp.groupby(by='hh')[['lny_h', 'lny_l']].cov()
timevar.reset_index(inplace=True)
blu = timevar.groupby(by='level_1').mean()
sd_yh = np.sqrt(blu.iloc[0,1])
sd_yl = np.sqrt(blu.iloc[1,2])

cov_yhyl = blu.iloc[1,1]/(sd_yh*sd_yl)
list_moments = [ sd_yh, sd_yl, cov_yhyl]



if save == True:
    print('datasets saved')
    data_sp.to_csv(folder+'CM_solution_10periods.csv.gz',compression='gzip')
    data_1t.to_csv(folder+'CM_solution_stationary.csv',index=False)
    


















