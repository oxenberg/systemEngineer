# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:44:38 2020

@author: oxenb
"""
import pandas as pd 
import scipy.stats

def ReadMeasuers():
    """read measures from file
        
        ----------
         
        Returns
        -------
        measuers_input {dataframe}
    """
    #read measuers_input
    measuers_input = pd.read_csv('../data/results/measuers.csv')
    return measuers_input

def calcStatistics():
    """sum all the cv over the mueasuers values and do a fridman test
        
        ----------
         
        Returns
        -------
        result {list}  statistic - The test statistic, correcting for ties, P value
    """
    measuers_input = ReadMeasuers()
    measuers_input = measuers_input[["AlgoName","Dataset_Name","AUC"]]
    measuers_input = measuers_input.groupby(["AlgoName","Dataset_Name"]).mean().reset_index()
    measuers_input = measuers_input[["AlgoName","AUC"]].groupby("AlgoName")['AUC'].apply(list)
    
    values = measuers_input.values
    KTboost = values[0]
    NGBclassifier = values[1]
    RF_baseClassifer = values[2]
    infiboost = values[3]
    result = scipy.stats.friedmanchisquare(KTboost,NGBclassifier,RF_baseClassifer,infiboost)
    return result

