# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:08:54 2019

@author: Albert Rodriguez Sala
"""
import numpy as np
import itertools

'''
File to contain different integration methods for Large-Scale Dynamic Economic Problems.
I make us of Matlab code provided in Serguei Maliar's website,
 of python code available in the quantecon site by S.Lyon and C.Coleman,
 and

References
==========
Lilia Maliar and Serguei Maliar, (2014).
"Numerical Methods for Large Scale Dynamic Economic Models” 
 Handbook of Computational Economics, Volume 3, Chapter 7.
 
Kenneth L. Judd, Lilia Maliar, Serguei Maliar and Rafael Valero, (2014).
 “Smolyak Method for Solving Dynamic Economic Models: Lagrange Interpolation,
 Anisotropic Grid and Adaptive Domain”, 
 Journal of Economic Dynamic and Control 44(C), 92-123.
 
 Chase Coleman, Spencer Lyon, Lilia Maliar and Serguei Maliar, (2018). 
 "Matlab, Python, Julia: What to Choose in Economics? CEPR working paper DP 13210
'''


def monomial_rule_m1(n,Sigma):
    '''
    function that constructs integration nodes and weights 
     under N-dimensional monomial (non-product) integration rule with 2N nodes. 
     -------------------------------------------------------------------------
     Inputs:  
         "n" is the number of random variables; N>=1;
         "Sigma" is the variance-covariance matrix; N-by-N

     Outputs: 
         "n_nodes" is the total number of integration nodes; 2*N;
         "ϵj" are the integration nodes; n_nodes-by-N;
         "ωj" are the integration weights; n_nodes-by-1
    '''
    assert Sigma.shape[0] == Sigma.shape[1], "Variance covariance matrix must be square"
    n_nodes = 2*n

    z1 = np.zeros((n_nodes, n))

    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i in range(n):
        z1[2*i:2*(i+1), i] = [1, -1]

    sqrt_Sigma = np.linalg.cholesky(Sigma)
    R = np.sqrt(n)*sqrt_Sigma
    ϵj = z1 @ R
    ωj = np.ones(n_nodes) / n_nodes

    return n_nodes, ϵj, ωj


def monomial_rule_m2(n, Sigma):
    '''
    function that constructs integration nodes and weights 
     under N-dimensional monomial (non-product) integration rule with 2*n^2+1 nodes. 
    -------------------------------------------------------------------------
    Inputs:  "n" is the number of random variables; n>=1;
              "Sigma" is the variance-covariance matrix; N-by-n

    Outputs: "n_nodes" is the total number of integration nodes; 2*n^2+1;
              "ϵj" are the integration nodes; n_nodes-by-n;
              "ωj" are the integration weights; n_nodes-by-1
    '''
    assert Sigma.shape[0] == Sigma.shape[1], "Variance covariance matrix must be square"
    n_nodes = 2*n**2+1
    z0 = np.zeros((1, n))

    z1 = np.zeros((2*n, n))
    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i in range(n):
        z1[2*i:2*(i+1), i] = [1, -1]

    z2 = np.zeros((2*n*(n-1), n))
    i = 0

    # In each node, a pair of random variables (p,q) takes either values
    # (1,1) or (1,-1) or (-1,1) or (-1,-1), and all other variables take
    # value 0. For example, for N = 2, `z2 = [1 1; 1 -1; -1 1; -1 1]`
    for p in range(n-1):
        for q in range(p+1, n):
            z2[4*i:4*(i+1), p] = [1, -1, 1, -1]
            z2[4*i:4*(i+1), q] = [1, 1, -1, -1]
            i += 1

    sqrt_Sigma = np.linalg.cholesky(Sigma)
    R = np.sqrt(n+2)*sqrt_Sigma
    S = np.sqrt((n+2)/2)*sqrt_Sigma
    ϵj = np.row_stack([z0, z1 @ R, z2 @ S])

    ωj = np.concatenate([2/(n+2) * np.ones(z0.shape[0]),
                         (4-n)/(2*(n+2)**2) * np.ones(z1.shape[0]),
                         1/(n+2)**2 * np.ones(z2.shape[0])])
    return n_nodes, ϵj, ωj



def gauss_hermite_quadrature(n_pol, mu, Sigma):
    '''
    function that constructs integration nodes and weights   
    under Gauss-Hermite quadrature (product) integration rule.
    
    # Inputs:  
        "n_pol" (int): is the number of nodes in each dimension 1<=n_pol<=100
        "mu" (array or list. Shape: N): is the vector of means of the vector of exogenous variables.
        "Sigma" (array or list. shape: NxN): is the variance-covariance matrix.

    # Outputs: 
        "n_nodes" is the total number of integration nodes n_pol^N
        "epsi_nodes" are the integration nodes n_nodes-by-n
        "weight_nodes" are the integration weights n_nodes-by-1
    '''  
 
    mu = np.array(mu)
    Sigma =np.array(Sigma)
    N = len(mu)
    const = np.pi**(-0.5*N)
    
     # Computes the sample points and weights for unidimensional Gauss-Hermite
    x, w = np.polynomial.hermite.hermgauss(n_pol)                           
    
    #Construct tensor-product grid 
    xn = np.array(list(itertools.product(*(x,)*N)))
    ## Delivers the grid points. len(rows(xn)) = number of grid points. ex: if polynomial 2, 2d --> 4. If polynomial 3, 2d --> 9
    
    wn = np.prod(np.array(list(itertools.product(*(w,)*N))), 1)
    ## The weight for each grid point.
    
    ## Change of variable in case of correlation of variables using cholesky decomposition.
    #Also to normalize weights so they sum up to 1 (as in Maliar and Maliar(2014)).    
    ϵj = 2.0**0.5*np.dot(np.linalg.cholesky(Sigma), xn.T).T + mu[None, :]
    ωj = wn*const
    n_nodes = len(ωj)
    return n_nodes, ϵj, ωj





def gauss_hermite_1d(n_pol, mu, sigma2):
    '''
    function that constructs integration nodes and weights   
    under Gauss-Hermite quadrature (product) integration rule.
    
    # Inputs:  
        "n_pol" (int): is the number of nodes in each dimension 1<=n_pol<=100
        "mu" (array or list. Shape: N): is the vector of means of the vector of exogenous variables.
        "Sigma" (array or list. shape: NxN): is the variance-covariance matrix.

    # Outputs: 
        "n_nodes" is the total number of integration nodes n_pol^N
        "epsi_nodes" are the integration nodes n_nodes-by-n
        "weight_nodes" are the integration weights n_nodes-by-1
    '''  
 
    mu = np.array(mu)
    sigma2 =np.array(sigma2)
    
    const = np.pi**(-0.5)
    
     # Computes the sample points and weights for unidimensional Gauss-Hermite
    x, w = np.polynomial.hermite.hermgauss(n_pol)                           
    
    
    ## Change of variable in case of correlation of variables using cholesky decomposition.
    #Also to normalize weights so they sum up to 1 (as in Maliar and Maliar(2014)).    
    ϵj = 2.0**0.5*sigma2**0.5*x +mu
    ωj = w*const
    n_nodes = len(ωj)
    return n_nodes, ϵj, ωj





#### FROM Coleman, Lyon, Maliar, Maliar (2017)
# https://notes.quantecon.org/submission/5b53cdaf17fb4900153deaff

def qnwmonomial1(vcv):
    n = vcv.shape[0]
    n_nodes = 2*n

    z1 = np.zeros((n_nodes, n))

    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i in range(n):
        z1[2*i:2*(i+1), i] = [1, -1]

    sqrt_vcv = np.linalg.cholesky(vcv)
    R = np.sqrt(n)*sqrt_vcv
    ϵj = z1 @ R
    ωj = np.ones(n_nodes) / n_nodes

    return ϵj, ωj


def qnwmonomial2(vcv):
    n = vcv.shape[0]
    assert n == vcv.shape[1], "Variance covariance matrix must be square"
    z0 = np.zeros((1, n))

    z1 = np.zeros((2*n, n))
    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i in range(n):
        z1[2*i:2*(i+1), i] = [1, -1]

    z2 = np.zeros((2*n*(n-1), n))
    i = 0

    # In each node, a pair of random variables (p,q) takes either values
    # (1,1) or (1,-1) or (-1,1) or (-1,-1), and all other variables take
    # value 0. For example, for N = 2, `z2 = [1 1; 1 -1; -1 1; -1 1]`
    for p in range(n-1):
        for q in range(p+1, n):
            z2[4*i:4*(i+1), p] = [1, -1, 1, -1]
            z2[4*i:4*(i+1), q] = [1, 1, -1, -1]
            i += 1

    sqrt_vcv = np.linalg.cholesky(vcv)
    R = np.sqrt(n+2)*sqrt_vcv
    S = np.sqrt((n+2)/2)*sqrt_vcv
    ϵj = np.row_stack([z0, z1 @ R, z2 @ S])

    ωj = np.concatenate([2/(n+2) * np.ones(z0.shape[0]),
                         (4-n)/(2*(n+2)**2) * np.ones(z1.shape[0]),
                         1/(n+2)**2 * np.ones(z2.shape[0])])
    return ϵj, ωj