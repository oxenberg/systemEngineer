# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:21:50 2020

@author: oxenb
"""

import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(train,test):
    #concate data
    train['dataType'] = 'train'
    test['dataType'] = 'test'
    test["CLASS"] = ""
    labelColumns = ["CLASS","dataType"]
    data = pd.concat([train, test])
    
    #handle missing value
    n = data.notnull()
    precentNotNull = 0.9
    data = data.loc[:, n.mean() > precentNotNull]
    
    catCol = []
    numCol = []
    binCol = []
    for col in data.columns[:-2]:
        if int(col[1:]) <= 83:
            catCol.append(col)
        else:
            if np.isin(data[col].dropna().unique(), [0, 1]).all():
                binCol.append(col)
            else:
                numCol.append(col)
    
    #most freq missing values
    imp_frq = SimpleImputer(strategy='most_frequent')
    imp_frq.fit(data[catCol+binCol+labelColumns])
    data_frq = pd.DataFrame( imp_frq.transform(data[catCol+binCol+labelColumns]), columns =catCol+binCol+labelColumns)
    
    mean_frq = SimpleImputer(strategy='mean')
    mean_frq.fit(data[numCol])
    data_mean = pd.DataFrame( mean_frq.transform(data[numCol]), columns =numCol)
    
    data = pd.concat([data_mean,data_frq],axis = 1)
    
    #sort class column to the end
    features = list(data.columns)
    features = [ elem for elem in features if elem not in labelColumns] 
    
    df_ohe_features = pd.get_dummies(data[features],prefix = catCol ,columns = catCol )

    #all dataframe to numric
    df_ohe_features = df_ohe_features.apply(pd.to_numeric)
    
    #normalize
    SC = StandardScaler()
    col = list(df_ohe_features.columns.values)
    df_ohe_features_val = SC.fit_transform(df_ohe_features.values)
    df_ohe_features = pd.DataFrame(df_ohe_features_val,columns = df_ohe_features.columns)
    df_ohe = pd.concat([df_ohe_features,data[labelColumns]],axis = 1)
    
    #split to test and train again
    train_processed = df_ohe[df_ohe['dataType']=='train']
    test_processed = df_ohe[df_ohe['dataType']=='test']
    
    #clean additonal columns
    test_processed = test_processed.drop(labelColumns,axis = 1)
    train_processed = train_processed.drop(["dataType"],axis = 1)
    
    #sort columns 
    train_processed,test_processed =  train_processed.align(test_processed, join = 'left', axis = 1)
    test_processed.drop('CLASS', inplace = True, axis = 1)
    return train_processed,test_processed

# --------- load the data ---------
missing_values = ["n/a", "Nane", "nan","Null"]

train = pd.read_csv('data/train.CSV',na_values = missing_values)
test = pd.read_csv('data/test.CSV',na_values = missing_values)

train,test = preprocess(train,test)


    
classCounter = train["CLASS"].value_counts()
numberToAdd = abs(classCounter["Yes"] - classCounter["No"])
train = train.append(train[train["CLASS"] == classCounter.argmin()].sample(n=numberToAdd, random_state=1),ignore_index = True)


train["CLASS"].value_counts()



