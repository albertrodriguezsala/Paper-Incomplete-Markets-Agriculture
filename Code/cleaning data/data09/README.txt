
Order on running the files

cons09, agric09, labor_bs09, sociodem09, wealth09, data09

Description of the files:

cons09.py
# =============================================================================
# Consumption Uganda 2009-10
# =============================================================================
'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2009 (ISA-LSMS) and computes:
         - food consumption prices at different regional levels. To use to value production.
         - Consumption dataset at the household level.
Main outcome: cons09.csv
'''

agric09.py
# =============================================================================
# #### Agricultural data Uganda 2009-10
# =============================================================================

'''
DESCRIPTION:
    Uses the data from the 2009-2010 Integrated Survey in Agriculture from the UNPS (ISA-LSMS) to obtain:
        For the 2 main harvest of the year of the survey. (fall 2009 and spring 2010/)
            -agricultural inputs variables at plot level and household level.
            -agricultural ouptut variables at plot level, household level, and crop level.   
        -household land, household farming assets (capital) household level
        - Livestock revenues (including non-sold consumption), costs, and stock (wealth) household level
     output: inc_agsec09.csv (agric income and costs at hh level) wealth_agrls09.csv (agric wealth: livestock+farming capital), agric_data09.csv (outputs and inputs at crop-plot-hh level)
'''



labor_bs09.py
# =============================================================================
# Labor and business income Uganda 2009-10
# =============================================================================
'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2009 (ISA-LSMS) and computes:
         - household labor income, labor supply, business proftis, and other sources of income
Main outcome: income_hhsec_2009.csv
'''


sociodem09.py
# =============================================================================
# Sociodemographic characteristics Uganda 2009-10
# =============================================================================
'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2009 (ISA-LSMS) and computes:
         - household sociodemographic statistics as gender,education, age, ethnicity, parental backround, etc.
Main outcome: income_hhsec_2009.csv
'''


wealth09.py

# =============================================================================
### Household Wealth: Uganda 2009-10
# =============================================================================
'''
DESCRIPTION
     Uses the data from the household questionnaire in the UNPS 2009 (ISA-LSMS) and computes:
         - household assets values.
    Uploads the cleaned data from the agricultural part and sums livestock and farm capital wealth
    Uploads data on land value (created from the surveys in a separate .py)
Main outcome: income_hhsec_2009.csv
'''

data09.py

# =============================================================================
#  DATA 2009-10 WAVE
# =============================================================================

'''
DESCRIPTION
    -  Merge the previously cleaned datasets on agriculture, consumption, income, wealth, labor and business income, sociodemographic characteristics.
Also adds basic information variables from the household survey (as country, region, urban, etc)
   - Computes the consumption, income, and wealth at the household level.
   - deflates monetary variables with the CPI index from the worldbank (entire country) and converts them to US 2013 dollars.
   - trims the consumption, income at wealth for extreme outliers. trimming level from 2.5 to 0.5 depending on the variable.
   - Provides summary statistics of consumption, income, and wealth for the wave 2009.
Creates the 2 household level datasets for the wave 2009: data09.csv (entire country and data09_rural.csv (only rural)
'''
