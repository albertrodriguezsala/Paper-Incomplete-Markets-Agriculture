# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:13:18 2024

@author: arodrig4
"""


import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import sys
dirct  = Path('Master_quantmodel.py').resolve().parent
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)


my_wd1 = dirct+'/quant model/'
print(my_wd1)

sys.path.append(my_wd1)

## (1) solve the benchmark economy under calibrated parameters. 
## to check calibration, open directly model_class.py, and solve_IM.py
# model_class.py is a python class that initializes the model. parameter values, grids, basic 
# functions are defined there.
# to change parameters I recommend to  initialize the household problem in solve_IM  model_class() with different parameter values.
# with other advantatges, the class object allows to have default params values, so changed in param values at initializing problem
# in solve_IM will not change the values used at solving CM or checking the data.
# alternatively one can change the values in model_class.py. Note that this changes the default values so it will change the values
# at solving IM (benchmark economy) but also at comparing model vs. data, at solving CM, and at the sensityivity analysis.

print('===========================================================================')
print('(1) SOLVING THE BENCHMARK ECONOMY, THE MAIN MODEL: INCOMPLETE MARKETS (IM)')
print('===========================================================================')
print(' ')
print('----------------------')
with open(my_wd1+'solve_IM.py', 'r') as file:
        code = file.read()
        exec(code)


# given the solution of the calibrated benchmark economy, let's compare it with some key statistics
# in the panel data of rural Uganda.
print('===========================================================================')
print('MODEL VS. DATA.')
print('===========================================================================')
print(' ')
print('----------------------')

with open(my_wd1+'model_vs_data.py', 'r') as file:
        code = file.read()
        exec(code)


# now let's solve the ideal scenario with complete markets in agriculture.
# it takes the default parameter values in the class model_class.py
print('===========================================================================')
print('(2) SOLVING COUNTERFACTUAL SP: COMPLETE MARKETS IN AGRICULTURE (CM)')
print('===========================================================================')
print(' ')
print('----------------------')

print(os.getcwd())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)

    with open(my_wd1+'solve_CM.py', 'r') as file:
        code = file.read()
        exec(code)
   
# this code runs the main results of the paper: the quantitative comparison of IM vs CM in agriculture in rural Uganda.
print('===========================================================================')
print('(3) MAIN RESULTS: AGGREGATE IMPACT INCOMPLETE FINANCIAL MARKETS IN AGRICULTURE.')
print('IM VS. CM')
print('===========================================================================')
print(' ')
print('----------------------')

with open(my_wd1+'IM_vs_CM_quantresults.py', 'r') as file:
        code = file.read()
        exec(code)

# This code obtains the efficiency gains of completing markets in agriculture.
# idea: constrain SP to maximize welfare st input usage same as benchmark but allocations of intermediates
# across heterog farmers and technologies is optimal (same as CM).
print('===========================================================================')
print('THE COST OF MISSALLOCATION, EFFICIENCY GAINS')
print('SP CONSTRAINED TO SAME LEVEL INPUTS BENCHMARK ')
print('CM CONSTRAINED  VS CM')
print('===========================================================================')
print(' ')
print('----------------------')

with open(my_wd1+'solve_CM_fixed_m.py', 'r') as file:
        code = file.read()
        exec(code)

