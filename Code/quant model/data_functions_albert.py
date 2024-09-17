# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:34:27 2018

@author: Albert
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outliers(df,lq=0,hq=1):
    #df: Dataframe with only the variables to trim
    # lq: lowest quantile. hq:Highest quantile
    columns = pd.Series(df.columns.values).tolist()
    for serie in columns:
        df["houtliers_"+serie] = df[serie].quantile(hq)
        df[df[serie]>df["houtliers_"+serie]] = np.nan
        df["loutliers_"+serie] = df[serie].quantile(lq)
        df[df[serie]<df["loutliers_"+serie]]= np.nan
        del df["houtliers_"+serie], df["loutliers_"+serie]
    return df


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



def data_stats(data, percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]):
    ### Compute mean, sd, min, max, quintiles and Gini of the states in the state list.
    # Inputs -----------------
    # Dataframe
    
    #Obtain the describtive statistics for the state variables.    
        
    summary = data.describe(percentiles)
    summary.reset_index(inplace=True)
        
    #Generate Gini statistic:
    gini_stat = np.empty(len(data.columns)).reshape(1,len(data.columns))
    
    for i, var in enumerate(data.columns):
        gini_stat[:,i] = gini(data[var].dropna())
                
    data_gini = pd.DataFrame(gini_stat, columns=data.columns)    
    data_gini.reset_index(inplace=True)
    data_gini['index'] = 'Gini'
    summary = summary.append(data_gini, ignore_index=True)
    summary.set_index('index', inplace=True)
    
    return summary


def plot_cond_log_distr(data, variable1, variable2, folder='C:/Users/rodri/Dropbox/JMP/', save=False):
        fig, ax = plt.subplots()
        a = data[variable2].unique()
        for value in a:           
            sns.distplot((np.log(data.loc[data[variable2] == value][variable1]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[variable1]).replace([-np.inf, np.inf], np.nan)).dropna()), label=variable2+str(value))
           
        plt.title('Distribution of '+variable1+' in Uganda')
        plt.xlabel(variable1)
        ax.legend()
        if save == True:
            fig.savefig(folder+'distr'+variable1+variable2+'.png')
            return plt.show()
        
def plot_cum_cond_log_distr(data, variable1, variable2, folder='C:/Users/rodri/Dropbox/JMP/', save=False):
        fig, ax = plt.subplots()
        a = data[variable2].unique()
        for value in a:           
            sns.distplot((np.log(data.loc[data[variable2] == value][variable1]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[variable1]).replace([-np.inf, np.inf], np.nan)).dropna()), label=variable2+str(value), hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
        plt.title('Cumulative Distribution of '+variable1+' in Uganda')
        plt.xlabel(variable1)
        ax.legend()
        if save == True:
            fig.savefig(folder+'distr'+variable1+variable2+'.png')
            return plt.show()       


def data_stats_2(data, key, quantiles=[0, 0.05, 0.25, 0.5, 0.75, 0.99, 1]):
        ### Compute mean, sd, min, max, quintiles and Gini of the states in the state list.
        # Inputs -----------------
        # State_list: List of state arrays [Nx1]
        #State Names: List of names of each state.
    
        #Obtain the describtive statistics for the state variables.    

        describe = data.describe().iloc[1:,]
        describe.reset_index(inplace=True)
        mean_sd = describe.iloc[0:2,:]
        
        pct = data[key].quantile(quantiles)
        pct=np.array(pct)
 
            # sort data (lowest to highest income)   
        datasort=data.sort_values(key)
    
        quant_data = np.zeros((len(quantiles), len(data.columns)))
        q=0
        for q in range(1, len(quantiles)):
            quant_data[q,:] = np.mean( datasort.loc[(datasort[key] > pct[q-1]) & (datasort[key] <= pct[q]) ] , axis=0) 
        
      
        
            
        gini_stat = np.empty(len(data.columns)).reshape(1,len(data.columns))
    
        for i, var in enumerate(data.columns):
            gini_stat[:,i] = gini(data[var].dropna())
            
                
        data_gini = pd.DataFrame(gini_stat, columns=data.columns)    
        data_gini.reset_index(inplace=True)
        data_gini['index'] = 'Gini'
        
        data_quant = pd.DataFrame(quant_data, columns=data.columns) 
        data_quant.reset_index(inplace=True)
        data_quant['index'] = quantiles
        summary = mean_sd.append(data_quant, ignore_index=True)
        summary = summary.append(data_gini, ignore_index=True)
    
        return summary


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist
