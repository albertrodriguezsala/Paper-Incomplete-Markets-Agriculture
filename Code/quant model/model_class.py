
"""
Created on Thu Sep 10 15:16:58 2020

@author: rodri
"""

# =============================================================================
# Class of the HH problem 
# =============================================================================

import numpy as np
import os
import quantecon as qe
import warnings
warnings.simplefilter('ignore')

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_data.py').resolve().parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

dirct = dirct+'/quant model/'

from integration_methods import gauss_hermite_quadrature, gauss_hermite_1d
import math
np.random.seed(23)

'''
    avg(y) = 1502.58
    avg(w) = = 1862
    avg(c)= 1500
    avg(y_na) = 495.51
    avg(y_h)= 651.12
    avg(y_l)= 353.39
    avg(m_h)= 40.75
    avg(m_l)= 36.46 
    gini(inc) = 0.57
    gini(y_a) = 0.62    
    gini(w) = 0.71
    risk y_h = 1.0521
    risk y_l = 0.722
    corr(y_h,y_l) = 0.115
    risk inc = 0.6004

'''


def my_lin(lb, ub, steps, spacing=2):
        span = (ub-lb)
        dx = 1.0 / (steps-1)
        return np.array([lb + (i*dx)**spacing*span for i in range(steps)])
    
    

class HH_model:
    
    ### Import model parameters, grids, functions.
        
    def __init__(self,  A = 276, alpha = 0.4, B =180, gamma = 0.4, C=0, D=0, rho =2, bbeta = 0.96, z=0,
                 spacing=1, N_x = 60, N_a =60, N_m1=15, N_m2=15, N_z=5, N_yna=5,
                  b= 0, m1_min=0.01, m2_min=0.01, x_min=10, factor_mh=1.5, factor_ml=1.5, a_max=18000,  x_max=25000, 
                  r = -0.1055, p = 30.7,  N=1000000, T=120, tol_error= 0.0002,
                  sig_theta= 1.0325, sig_eps=0.853, corr=0.108, N_pol=7,  #sigmas (no labor) = 1.068, 0875
                   mu_yna=np.log(497), rho_yna= 0.3994, sig_yna=1.33, # rho_hat=0.39. sigma_hat=1.33
                  mu_z=0, sig_z=0.277):
                 
        
        self.A, self.B, self.alpha , self.gamma,  self.rho, self.bbeta = A, B, alpha, gamma, rho, bbeta 
        self.N_x, self.N_a, self.N_m1, self.N_m2 =N_x, N_a, N_m1, N_m2
        self.spacing =  spacing
        SIGMA = np.array([[sig_theta**2,corr],[corr,sig_eps**2]])
        self.SIGMA, self.sig_theta, self.sig_eps, self.sig_z, self.corr_shocks = SIGMA, sig_theta, sig_eps, sig_z, corr
        self.r = r
        self.p = p
        self.b = b
        self.a_max = a_max
        self.N = N
        self.T = T
        self.tol_error = tol_error
        self.N_yna, self.rho_yna, self.sig_yna = N_yna, rho_yna, sig_yna
        self.N_z,  self.sig_z = N_z,  sig_z
        self.C = C
        self.D = D
        self.mu_yna = mu_yna
        
        p_z = N_z*[1/N_z]
        ### Shocks Integration: 7-nodes Gauss-Hermite quadrature rule
        n_nodes, ghq_nodes, w_j = gauss_hermite_quadrature(n_pol=N_pol, mu=[0,0], Sigma=SIGMA)
        theta_j, eps_j = np.exp(ghq_nodes[:,0]-sig_theta**2/2), np.exp(ghq_nodes[:,1]-sig_eps**2/2)
        self.theta_j, self.eps_j, self.w_j = theta_j, eps_j, w_j

    
        
        ### Markov process for the AR(1) shock process
        mc_yna  = qe.rouwenhorst( mu= mu_yna*(1-rho_yna), rho= rho_yna, sigma= sig_yna, n= N_yna)
        pi_yna_star = mc_yna.stationary_distributions
        yna_grid = np.exp(mc_yna.state_values)
        exp_yna = pi_yna_star@yna_grid
        yna_grid = (yna_grid / exp_yna)*np.exp(mu_yna)
        pi_yna = mc_yna.P
        self.yna_grid, self.pi_yna, self.pi_yna_star = yna_grid, pi_yna, pi_yna_star
        
        
        ### Permanent productivity
        ## hugguett-parra(10, JPE) for perm prod use 5 evenly spaced points on [mu-3sigma, mu+3sigma]
        z_min         = -sig_z**2 -3*sig_z
        z_max         = +sig_z**2 +3*sig_z
        z_grid        = np.linspace(z_min,z_max,N_z)
        p_z           = N_z*[1/N_z]
        z_grid        = np.exp(z_grid)
        exp_z         =  p_z @ z_grid 
        self.z_grid   = z_grid/exp_z
        self.p_z      = p_z
        
        
        n_nodes, z_nodes, w_z = gauss_hermite_1d(n_pol=N_z, mu=[0], sigma2=sig_z**2)
        z_j = np.exp(z_nodes-sig_z**2/2)
        #self.z_grid   = z_j
        #self.p_z      = w_z
        
        
        mh_star = (p/(z_grid[N_z-1]*A*bbeta*alpha*C))**(1/(alpha-1))
        ml_star =  (p/(z_grid[N_z-1]*B*bbeta*gamma*D))**(1/(gamma-1))
        
        
        m1_max, m2_max = factor_mh*mh_star, factor_ml*ml_star
        self.m1_max, self.m2_max = m1_max, m2_max
        self.m1_min,self.m2_min = m1_min, m2_min
           
        ### Initialize grids:
        #x_max = 80000  #60000*(1+r) +A*m1_max**alpha +B*m2_max**gamma +np.max(z_j)
        self.x_min, self.x_max = x_min, x_max
        #x_min = b +A*0.1**alpha +B*0.1**gamma +np.min(yna_grid)
        x_grid = my_lin(x_min, x_max, N_x, spacing=spacing)
        self.x_grid = x_grid
        a_grid = my_lin(b+1e-2, a_max, N_a, spacing=spacing)
        self.a_grid = a_grid
        m1_grid = my_lin(m1_min, m1_max, N_m1, spacing=spacing) 
        self.m1_grid = m1_grid
        m2_grid = my_lin(m2_min, m2_max, N_m2,spacing=spacing)
        self.m2_grid = m2_grid
        


        
### Utility functions ----------------   
    def u_cont(self,c):
        return ((c)**(1-self.rho))/(1-self.rho) 
    
        
# Production functions ----------    
    def y1(self,z,m1,θ):
        return z*θ*self.A*math.pow(m1,self.alpha)+self.C
     
    def y2(self,z,m2,ε): 
        return z*ε*self.B*math.pow(m2,self.gamma)+self.D 
            
       
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
    
   
    

