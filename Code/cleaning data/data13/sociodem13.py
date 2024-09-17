# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:30:29 2018

@author: Albert
"""

# =============================================================================
# Sociodemographic characteristics Uganda 2013-14
# =============================================================================

'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2013-14 (ISA-LSMS) and computes:
         - household sociodemographic statistics as gender,education, age, ethnicity, parental backround, etc.
Main outcome: sociodem_2013.csv
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
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

pd.options.display.float_format = '{:,.2f}'.format

# =============================================================================
#  Merging Sociodemographic characteristics 2013-14
# =============================================================================

#Age
age = pd.read_csv(my_dirct+'gsec2.csv')
age = age[["HHID","PID","h2q3","h2q4", "h2q5", "h2q6","h2q8"]]
age.columns = ["hh","pid","sex", 'hh_member',"months_in_hh","reason_leave", "age"]

#Background, bednets and migration
bck = pd.read_csv(my_dirct+'gsec3.csv')
bck = bck[["PID","h3q3","h3q4", "h3q9","h3q10"]]
bck.columns = ["pid","father_educ", "father_ocup","ethnic","bednet"]
bck.father_educ = bck.father_educ.replace(99,np.nan)
#Group bednet answer as yes I have, no 
bck.bednet = bck.bednet.replace([2 , 3, 9],[1 , 0, np.nan])


#Education
educ = pd.read_stata(my_dirct+'GSEC4.dta', convert_categoricals=False)
educ = educ[["PID","h4q4", "h4q7"]]
educ.columns = ["pid","writeread","classeduc"]
educ.loc[educ["classeduc"]==99, "classeduc"] = np.nan
educ.writeread = educ.writeread.replace([2, 4, 5],0)
#1 if able to read and write. 0 if unable both, unable writing, uses braille

### Merge for household head characteristics===================================
socio = pd.merge(age, bck, on="pid", how="inner")
socio = pd.merge(socio, educ, on="pid", how="inner")
socio = socio.loc[(socio.hh_member==1)]
socio.drop(["hh_member", "pid", "months_in_hh","reason_leave"], axis=1, inplace=True)

socio['hh'].value_counts()


##### Household characteristics ###################################

## familysize
familysize = age.groupby(by='hh').size().to_frame()
familysize.columns = ['familysize']
familysize.reset_index(inplace=True)


## Health
health = pd.read_csv(my_dirct+'gsec5.csv')
health = health[["PID","h5q4","h5q5", "h5q6"]]
health.columns = ["pid","illyes","illdays", "ill_stop_activity"]
health['illyes'].replace([1,2], [0,1], inplace=True)
health = health.merge(age, on="pid", how="inner")

# Sickness by adult and kids
health["kids_sick"] = health.loc[health.age<16, "illyes"]
health["adults_sick"] = health.loc[health.age>16, "illyes"]
health["kids_ill_days"] = health.loc[health.age<16, "illdays"]
health["adults_ill_days"] = health.loc[health.age>16, "illdays"]

health_hh = health.groupby(by='hh')[["illyes", "kids_sick","adults_sick","illdays", "ill_stop_activity", "kids_ill_days", "adults_ill_days" ]].sum()
health_hh.reset_index(inplace=True)


#welfare
#welf = pd.read_stata('GSEC17.dta', convert_categoricals=False)
#welf = welf[["HHID","h17q9","h17q10","h17q11", "h17q13"]]
#welf.columns = ["hh", "not_enough_food","not_enough_food_months","not_enough_food_reason", "ranout_food"]

#Sociodemographic dataset
socio = pd.merge(socio, familysize, on="hh", how="left")
socio = pd.merge(socio, health_hh, on="hh", how="left")
#socio = pd.merge(socio, welf, on="hh", how="left")


socio.to_csv(folder+"sociodem13.csv")