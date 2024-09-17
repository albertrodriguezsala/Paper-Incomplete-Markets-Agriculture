
Infrastructure

Code ran in Python 3.11.7, with Microsoft Visual Studio Code under Anaconda environment. The code must be run in Python 3.11.7 (or similar versions). With newer versions some of the functions used in cleaning the data are depreciated and will not work---delivering a code error.

The code might also be run in other Python IDEs, but it might require some changes in the code. For example, in Spyder the relative paths could not be recognized so usually one needs to add an extra .parent.

Besides using user-defined modules in scrips, I also make use of other modules that need to be installed in the environment before running the code: 

- install quantecon package: type conda install quantecon either in condas terminal or in VS code powershell. Make sure VS code is using the correct Python environment and is integrated in the anaconda environment.
- Install linearmodels package: conda install -c conda-forge linearmodels



Replication Folder Organization

- data (datasets): contains the raw data and stores all the clean data from the UNPS 2009/10-2015/16.
- cleaning data (code): Contains the code to clean the raw data and create the panel datasets.
- empirics data (code): contains the code to run the empirical analysis and compute the statistics for the calibration.
- quant model (code): contains the code that solves the model of the economy (incomplete markets), the counterfactual (complete markets) and the comparison of the model vs. data and the quantitative results (IM vs CM).
- Results: stores all the empirical results as well as the numerical results (figures, simulated datasets, arrays, etc.).



Code Organization

There are 2 Master files that take the raw data and generate all the results in the paper:
(1) Master_data.py: cleans the raw data and provides the empirical analysis. To do so, it calls the files in folders cleaning data and in empirics data.
In more detail, Master_data.py does
1.  For each wave, runs the files that clean consumption, agriculture, labor income, business income, sociodemographic characteristics and wealth. Also runs de file that combines the datasets from the previous runs into the one wave main dataset: dataWAVE.
2.  Creates panels: household level panel dataset and plot level panel datasets (for crop analysis).
3.  Runs the empirical analysis: crop risk vs yields, crop selection among farmers; descriptive statistics in agriculture, consumption income, and wealth; moments to target (or compare) in the calibration of the model.

(2) Master_quantmodel.py: solves the models and provides the quantitative results. Uses the files in folder (quant model).
In more detail, Master_quantmodel.py does:
1. Solve the benchmark economy under incomplete markets (IM) and provides the comparison of the model vs. data.
2. Solve the complete markets (CM) economy and deliver the main results of the paper by comparing the outcomes of the IM economy vs the CM economy.


Code files list

In folder cleaning data:
   auxiliary codes:
- kg_conversions.py.
- land_value_imputation.py
   data09:
- agric09.py
- cons09.py
- data09.py
- labor_bs09.py
- sociodem09.py
- wealth09.py
   data10:
- agric10.py
- cons10.py
- data10.py
- labor_bs10.py
- sociodem10.py
- wealth10.py
   data11:
- agric11.py
- cons11.py
- data11.py
- labor_bs11.py
- sociodem11.py
- wealth11.py
   data13:
- agric13.py
- cons13.py
- data13.py
- labor_bs13.py
- sociodem13.py
- wealth13.py
   data15:
- agric15.py
- cons15.py
- data15.py
- labor_bs15.py
- sociodem15.py
- wealth15.py

- panel.py
- panel_plotcrop_data.py


In folder empirics data:
- CIW_summary_avgmoments.py
- empirics_yieldrisk_crops.py
- empiricscrops_riskmoments.py


In folder quant model: 
- data_functions_albert.py
- fixed_point.py
- IM_vs_CM_quantresults.py
- integration_methods.py
- model_class.py
- model_vs_data.py
- solve_CM.py
- solve_CM_fixed_m.py
- solve_IM.py






