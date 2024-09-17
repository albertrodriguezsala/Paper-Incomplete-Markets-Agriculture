"""
Created on Mon Apr  8 11:00:57 2024

@author: rodri
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_data.py').resolve().parent
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_wd1 = dirct+'/cleaning data'
my_wd2 = dirct+'/empirics data'


## (1) Clean the data for each wave ==================================

# For each wave, runs the files that clean consumption, agriculture, labor and business income, sociodemographic characteristics and wealth.
# Also runs de file that combines together the datasets from the previous runs into the one wave main dataset: dataWAVE.

for wave in ['09','10','11','13','15']:
    print('===========================================================================')
    print('CLEANING WAVE 20'+wave)
    print('===========================================================================')
    print(' ')
    print('----------------------')
    print('Consumption 20'+wave)
    with open(my_wd1+'/data'+wave+'/cons'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
    
    
    print(' ')
    print('----------------------')
    print('Agriculture 20'+wave)
    print('(1) Creates dataset for household agricultural income  inputs, and wealth')
    print('(2) Creates the crop-plot level dataset for the empirical findings section.')
    with open(my_wd1+'/data'+wave+'/agric'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
    
    print(' ')
    print('----------------------')
    print('Non-agric Earnings 20'+wave)
    with open(my_wd1+'/data'+wave+'/labor_bs'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)

    print(' ')
    print('----------------------')
    print('Sociodemographic charaterstics 20'+wave)
    with open(my_wd1+'/data'+wave+'/sociodem'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
  
    
    print(' ')
    print('----------------------')
    print('Wealth 20'+wave)
    with open(my_wd1+'/data'+wave+'/wealth'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
  
    

    print(' ')
    print('----------------------')
    print('household dataset wave'+wave)
    with open(my_wd1+'/data'+wave+'/data'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
   
        
        
        
#%%(2) Create panels: household level and plot level (for crop analysis) ==================================

print('===========================================================================')
print('CREATES THE PANEL DATASETS')
print('===========================================================================')



print(' ')
print('----------------------')
print('Create Household panel, UNPS 2009-2015.')

text = ''''imports the datasets for each wave dataXX.csv and dataXX_rural.csv and outputs
      (1) unbalanced panel all Uganda: panel_UGA.csv
      (2) unbalanced panel rural Uganda: panel_rural_UGA.csv'''

print(text)
with open(my_wd1+'/panel.py', 'r') as file:
    code = file.read()
    exec(code)



print(' ')
print('----------------------')
print('Create Household panel, UNPS 2009-2015.')

text = ''''imports the datasets for agriculture in each wave and outputs
      (1)unbalanced panel plot-crop level: panel_plotcrop_data.csv'''

print(text)
with open(my_wd1+'/panel_plotcrop_data.py', 'r') as file:
    code = file.read()
    exec(code)



#%% (3) Run the results from the data   ============================================================================
# - crop vs yields vs crop selection empricial findings
# - descriptive statistics in the data
# - moments to target (or compare) the calibration of the model.


print('===========================================================================')
print('EMPIRICAL FINDINGS, DATA SUMMARIES, AND DATA MOMENTS')
print('===========================================================================')


print(' ')
print('----------------------')
print('Summary Consumption, Income, Wealth and Data Moments')

text = ''''Using panel_UGA.csv and panel_rural_UGA.csv
        (1) Computes key moments targeted and non-targeted.
        (2) Provides Table 3 (summary CIW) and the CIW tables in the appendix.
'''

print(text)
with open(my_wd2+'/ciw_summary_avgmoments.py', 'r') as file:
    code = file.read()
    exec(code)



print(' ')
print('----------------------')
print('Empirical finding (1): crops yields vs risk')


text = '''
Takes the panel plot-crop level data and produces

(1)
Figure 1: Crop Yields vs. Risk
Table 1: Robustness Crop Yields vs. Risk
Table 11 appendix: Shares Selling Crops
Table 1 and 2 in the online appendix

(2) Classifies crops in the High Crops vs. Low Crops based on Figure 1. 
Low Crops: Below Median. High Crops: Above Media
'''


print(text)
with open(my_wd2+'/empirics_yieldsrisk_crops.py', 'r') as file:
    code = file.read()
    exec(code)




print(' ')
print('----------------------')
print('Empirical findings (2), (3). Output, input, and risk moments on crops, consumption and income')


text = '''
Takes the panel plot-crop level data and produces
Tables:
    - Table 2 Percentage Growing Crops
    - Table 11 Agric Output to Market, Own Consumption, Stored, Gifts
    - Table 12 Intermediate Input Summaries
    - Table 10: Regression Share High vs Low Crops on CIW 

-Figure 2: Shares Crops Across Wealth Distribution
    
Estimates:
    - Estimates the AR(1) process on non-agricultural income
    - Estimates the volatility of the High Crops and Low Crops.
    - Estimates the volatility on consumption and income.

Data Moments:
    Targeted: Output High and Low Crops, Risk High Crops, Risk Low Crops, Input Usage Low High and Low Crops.
    Non-Targeted: Consumption Risk, Income Risk.
'''


print(text)
with open(my_wd2+'/empiricscrops_riskmoments.py', 'r') as file:
    code = file.read()
    exec(code)







    
