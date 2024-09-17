# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:44:33 2018

@author: Albert
"""

# =============================================================================
# Labor and business income Uganda 2013-14
# =============================================================================


'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2013-2014 (ISA-LSMS) and computes:
         - household labor income, labor supply, business proftis, and other sources of income
Main outcome: income_hhsec_2013.csv
'''

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

os.chdir(dirct)

my_dirct = str(dirct)+'/data/raw data/2013/'
folder =  str(dirct)+'/data/data13/'
folder2 = str(dirct)+'/data/auxiliary data/'
# To pass all monetary variables to US 2013 $
dollars = 2586.89

#%%Income 

lab9 = pd.read_stata(my_dirct+"GSEC8_1.dta")
lab9 = lab9[["HHID","PID","h8q30a","h8q30b", "h8q31a","h8q31b","h8q31c","h8q44","h8q44b","h8q45a","h8q45b","h8q45c", "h8q47"]]     
lab9.columns = ["hh","pid","months1","weeks1","cash1","inkind1", "time1", "months2","weeks2", "cash2","inkind2", "time2", "type_job"]

a = pd.value_counts(lab9.type_job)


#very few people report hours so omitt them
pd.value_counts(lab9.time1).reset_index()
pd.value_counts(lab9.time2).reset_index()

#Compute means to use in 2011-2012
avg_months = lab9.months1.mean()
avg_weeks = lab9.weeks1.mean()

lab9["pay1"] = lab9.loc[:,["cash1","inkind1"]].sum(axis=1)
lab9["pay2"] = lab9.loc[:,["cash2","inkind2"]].sum(axis=1)
del lab9["cash1"], lab9["inkind1"], lab9["cash2"], lab9["inkind2"]


#Creating week wages
# From Besamuscka, Tijdens (2012),  https://wageindicator.org/main/Wageindicatorfoundation/publications/2012/wages-in-uganda-wage-indicator-survey-2012
# we take mean average hours and days worked per week: 60 and 6
lab9.loc[lab9.time1 == "Day", 'pay1'] = lab9.loc[lab9.time1 == "Day", 'pay1']*6
lab9.loc[lab9.time1 == "Month", 'pay1'] = lab9.loc[lab9.time1 == "Month", 'pay1']/4
lab9.loc[lab9.time1 == "Hour", 'pay1'] = lab9.loc[lab9.time1 == "Hour", 'pay1']*60

lab9.loc[lab9.time1 == "Day", 'pay2'] = lab9.loc[lab9.time1 == "Day", 'pay2']*6
lab9.loc[lab9.time1 == "Month", 'pay2'] = lab9.loc[lab9.time1 == "Month", 'pay2']/4
lab9.loc[lab9.time1 == "Hour", 'pay2'] = lab9.loc[lab9.time1 == "Hour", 'pay2']*60

### Setting averages can substantially limit inequality measures

lab9["wage1"] = lab9.months1*lab9.weeks1*lab9.pay1
lab9["wage2"] = lab9.months2*lab9.weeks2*lab9.pay2



lab99 = lab9.groupby(by="hh")[["wage1","wage2"]].sum()

lab99["wage_total"] = lab99.loc[:,["wage1","wage2"]].sum(axis=1)
lab99 = lab99.replace(0, np.nan)

summaryw = lab99.describe()/dollars
print(summaryw)

del lab9

#%% business

bus12 = pd.read_stata(my_dirct+'gsec12.dta')
bus12 = bus12[["hhid","h12q12", "h12q13","h12q15","h12q16","h12q17"]]
bus12.rename(columns={'hhid':'hh'}, inplace=True)
bus12.rename(columns={'h12q13':'revenue'}, inplace=True)
bus12["cost"] = -bus12.loc[:,["h12q15","h12q16","h12q17"]].sum(axis=1)
bus12["bs_profit"] = bus12.loc[:,["revenue","cost"]].sum(axis=1)
bus12["bs_profit"] = bus12["bs_profit"].replace(0,np.nan)*bus12['h12q12']
bus12['bs_revenue'] = bus12['revenue'].replace(0,np.nan)*bus12['h12q12']
bus12 = bus12[["hh","bs_profit",'bs_revenue']]
bus12 = bus12.groupby(by="hh").sum()
bus12 = bus12.replace(0, np.nan)

summarybus = bus12.describe()/dollars

print(summarybus)

#%% Other income

other = pd.read_stata(my_dirct+'GSEC11A.dta')
other = other[["HHID","h11q2","h11q5","h11q6"]]
other.rename(columns={'HHID':'hh'}, inplace=True)
other["other_inc"] = other.loc[:,["h11q5","h11q6"]].sum(axis=1)
other['remitances_in'] = other.loc[other['h11q2']==42 |43, 'other_inc'] 

other = other[["hh","other_inc", "remitances_in"]]
other = other.groupby(by="hh").sum()


#%% Merge datasets
income_gsec = pd.merge(lab99, bus12, on="hh", how="outer")
income_gsec = pd.merge(income_gsec, other, on="hh", how="outer")
#del income_gsec["wage1"], income_gsec["wage2"], bus12, c5, extra, other, lab99, summarybus, summaryo, summaryw

sum_income = income_gsec.describe()/dollars
income_gsec.to_csv(folder+'income_hhsec13.csv')