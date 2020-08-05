#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt

import numpy as np
import joblib
import pandas as pd
from collections import OrderedDict
import random
import json

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,precision_score,average_precision_score
from sklearn.datasets import fetch_covtype, load_svmlight_file
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

from scipy import interp

import datetime
import sys
import os

import pathlib
from tqdm import tqdm
#ktboost model
import KTBoost.KTBoost as KTBoost

from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from logitboost import LogitBoost


global INSTANCES_TO_MEASURE_TIME
INSTANCES_TO_MEASURE_TIME = 1000

# In[5]:


sys.path.append("../infiniteboost/research/")
from SparseInfiniteBoosting import InfiniteBoosting



# In[8]:

def getsDataPaths():
    """get all the datasets paths for import
    
        
        Returns
        -------
        allDataSetsPaths : {list}
            all the paths to the datasets
            
    """
    allDataSetsPaths = []
    dataSetName = "classification_datasets"
    dataPath = "../data"
    for file in os.listdir(f"{dataPath}/{dataSetName}"):
        if file.endswith(".csv"):
            allDataSetsPaths.append(os.path.join(f"{dataPath}/{dataSetName}", file))
    return allDataSetsPaths

# In[9]:


def getBadLabel(data,TH = 10):
    """Find the labels with less then some threshold of instances
    
        ----------
        data : {dataframe} of shape (n_samples, n_features)
        
        TH : {int} the minimun value of instances with same label
        for exemple if we have less then TH rows with same label we will consider
        it bad label
        
        
        Returns
        -------
        badLabels : {indexs}
            all the labels that answer the condtion
            
    """
    countSeries = data.iloc[:,-1].value_counts()
    badLabels = countSeries[countSeries< TH].index
    return badLabels

def preprocess(path):
    """preprocess the datasets. read each dataset,impute missing values and convert categorial columns
    
        ----------
        path : {string} path to the dataset
        
    
        Returns
        -------
        X: {dataframe} of shape (n_samples, n_features)
        y: {dataframe} of shape (n_samples, 1) - label column
            
            
    """
    data = pd.read_csv(path)
    
    imp_frq = SimpleImputer(strategy='most_frequent')
    imp_frq.fit(data)
    data = pd.DataFrame( imp_frq.transform(data), columns =data.columns)
    
    data = data.dropna()
    
    badLabelTH = 10
    badLabels = getBadLabel(data,badLabelTH)
    data.iloc[:,-1] = data.iloc[:,-1].apply(lambda x : "other" if x in list(badLabels) else x)

    #check if we still have bad lables under the TH
    badLabels = getBadLabel(data,badLabelTH)
    data = data[~data.iloc[:,-1].isin(badLabels)]


    strCoulmns = data.dtypes[data.dtypes == "object"].index
    if len(strCoulmns) > 0:
        le = preprocessing.LabelEncoder()
        for i in strCoulmns:
            data[i] = data[i].astype('str')
            data[i] = le.fit_transform(data[i])
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    return X, y


def calcFitTime(RS_model, X_train, y_train):
    """calculate the time to train the model

        ----------
        RS_model : {sklearn model, other model with fit method} the model
        X_train,y_train : {array-like}  of shape (n_samples, n_features)

    
        Returns
        -------
        T_time: {datetime} 
            
            
    """
    now = datetime.datetime.now()
    RS_model.fit(X_train, y_train)
    time_stop = datetime.datetime.now()
    T_time = time_stop- now ##for time calc of training
    return T_time

def calcInferenceTime(X_test, RS_model):
    """calculate the time to test 1000 rows from the dataset the model and calculate the predict
       values

        ----------
        RS_model : {sklearn model, other model with fit method} the model
        X_test : {array-like}  of shape (n_samples, n_features)

    
        Returns
        -------
        inferenceTime: {datetime} 
        y_pred: {array-like} predict values
            
            
    """
    instancesOfTest = len(X_test)
    if (instancesOfTest) < INSTANCES_TO_MEASURE_TIME:
        now = datetime.datetime.now()
        y_pred = RS_model.predict(X_test)
        time_stop = datetime.datetime.now()
        inferenceTime = (time_stop- now) *(INSTANCES_TO_MEASURE_TIME/instancesOfTest)
    else:
        indexSample = random.sample(list(np.arange(len(X_test))), INSTANCES_TO_MEASURE_TIME)
        X_test_sample = X_test[indexSample]
        now = datetime.datetime.now()
        RS_model.predict(X_test_sample)
        time_stop = datetime.datetime.now()
        inferenceTime = time_stop- now
        y_pred = RS_model.predict(X_test)
    
    return inferenceTime, y_pred

def calcFprTpr(y_test, y_pred_proba, classes):
    """calculate the fpr and tpr with the macro average method

        ----------
        y_pred_proba,y_test : {array-like}  of shape (n_samples, n_features)
        classes : {int} amount of classes for the dataset
    
        Returns
        -------
        inferenceTime: {datetime} 
        y_pred: {array-like} predict values
            
            
    """
    fpr = dict()
    tpr = dict()
    
    for i,class_ in enumerate(classes):
        fpr[class_], tpr[class_], _ = roc_curve(y_test == class_, y_pred_proba[:, i])
    
    all_fpr = np.unique(np.concatenate([fpr[class_] for class_ in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for class_ in classes:
        mean_tpr += np.interp(all_fpr, fpr[class_], tpr[class_])
    
    # Finally average it and compute AUC
    mean_tpr /= len(classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    return tpr["macro"], fpr["macro"]

def calcAPS(y_test,y_pred_proba):
    """calculate the average precison score (area uneder the precison recall curve)

        ----------
        y_pred_proba,y_test : {array-like}  of shape (n_samples, n_features)
    
        Returns
        -------
        APS: {array-like} 
            
    """
    #trnasform y to mulitly label format (vector) for average_precision_score func
    enc = OneHotEncoder(handle_unknown='ignore')
    multiyY = enc.fit_transform(y_test.reshape(-1,1)).toarray()
    APS = average_precision_score(multiyY,y_pred_proba, average='macro')
    return APS
    
def clacMeasure(RS_model,X_train,X_test,y_train,y_test, datasetName,AlgoName = None, cvIndex = None, n_classes = 2, multiclass = False):
    """Calculating mesurements over all the datasets and algorithems
    
        ----------
        RS_model : {sklearn randomsearch model} the model we use 
        to calc the preformance on
        
        X_train,X_test,y_train,y_test : {array-like}  of shape (n_samples, n_features)
        
        datasetName,AlgoName : {string}
        
        cvIndex,n_classes : {int}
        
        multiclass : {bool}
        
        Returns
        -------
        row : {list} 
            contain all the needed measuerments
            
    """
    row = []
    
    T_time = calcFitTime(RS_model, X_train, y_train)
    
    inferenceTime, y_pred = calcInferenceTime(X_test, RS_model)
    y_pred_proba = RS_model.predict_proba(X_test)
    

    #collect measures
    row.append(datasetName)
    row.append(AlgoName)
    row.append(cvIndex)
    row.append(str(RS_model.best_params_))
    acc = accuracy_score(y_test,y_pred)
    row.append(acc)
  
       
    #soultion for multiclass
    
    tpr, fpr = calcFprTpr(y_test, y_pred_proba,n_classes)
    row.append(tpr)
    row.append(fpr)
    
    # calculating precsion
    precsion = precision_score(y_test, y_pred, average='macro')
    row.append(precsion)
    
    # calculating AUC
    if(multiclass):
        AUC = roc_auc_score(y_test, y_pred_proba,multi_class = "ovr", average = 'macro')
    else:
        AUC = roc_auc_score(y_test, y_pred_proba[:,1], average = 'macro')
    row.append(AUC)
    
    APS = calcAPS(y_test,y_pred_proba)
    
    row.append(APS)
    row.append(T_time)
    row.append(inferenceTime)
    return row
# In[10]:



# In[11]:
def CreateModels():

    np.random.seed(42)
    models = {}
    
    distributions = dict(estimator__n_estimators=np.arange(50,300,20),estimator__max_leaf_nodes = [2,4,5])
    models['infiboost'] = [InfiniteBoosting(),distributions]
    
    distributions = dict(estimator__max_depth=np.arange(3,10),estimator__n_estimators=np.arange(50,300,20))
    models['KTBoost'] = [KTBoost.BoostingClassifier(),distributions]
    
    distributions = dict(estimator__n_estimators = np.arange(50,300,20),estimator__Base__max_depth=np.arange(4,10,1))
    models["NGBClassifier"] = [NGBClassifier(Dist=Bernoulli),distributions]
    
    distributions = dict(estimator__n_estimators = np.arange(50,300,20),estimator__max_depth=np.arange(3,15,1))
    models["RF_baseClassfier"] = [RandomForestClassifier(),distributions]
    
    return models

# In[12]:


def RunModels(models,allDataSetsPaths):
    """run the model for each dataset and for each algorithm with cv and random search. 
       create the measure table with the clacMeasure function and export it to file.
        
    
        ----------
        models : {dictonary}  key: model name, value: {list} contain model and hyperparmater distributions
        
        allDataSetsPaths : {list}  all paths to datasets
        
        Returns
        -------

            
    """
    measuers = pd.DataFrame(columns = ["Dataset_Name","AlgoName","CrossVal","HP_vals",
                                        "ACC","TPR","FPR","Precsion","AUC","Precstion_Recall",
                                        "Training_Time","Inference_Time"])
    
    index = 0
    #run over all the algo
    for AlgoName,items in models.items():
        model = items[0]
        dist = items[1]
        print(f"run on {AlgoName}..")
        # run over all the datasets
        
        for datasetName in tqdm(allDataSetsPaths[:10]):
            X, y = preprocess(datasetName)
            X = X.values
            y = y.values
            multiclass = False
            if(len(np.unique(y))>2):
                    multiclass = True
            folder = StratifiedKFold(n_splits=TEST_TRAIN_CV, shuffle=True, random_state=42)
            cvIndex = 1
            # run over train test cv
            for train_indices, test_indices in folder.split(X, y):
                print(f"cv num {cvIndex}..")
                #create data after cv from indexs
                X_train = X[train_indices]
                X_test = X[test_indices]
                y_train = y[train_indices]
                y_test = y[test_indices]
                
                #the selected model in the iteration of the cv, we need to do 1 vs all for the multi class 
                # if is binary class we stil can use the 1 vs all becouse is act the same
                clf = OneVsRestClassifier(clone(model))
                # we chose for the hyperparm tuning random search
                RS = RandomizedSearchCV(clf, dist, random_state=42,n_iter = RANDOM_SEARCH_ITER,cv = TRAIN_VALIDATION_CV)
    
                #add row to the measuerment table
                classes = np.unique(y)
                Dataname = datasetName.split("\\")[1].split(".")[0]
                measuers.loc[index] = clacMeasure(RS,X_train,X_test,y_train,y_test, Dataname,AlgoName, cvIndex, classes, multiclass)
                cvIndex+=1
                index+=1
        #     del clf
    
    #save the measures
    measuers.to_csv("../data/results/measuers.csv")
    measuers.to_pickle('../data/results/measuers_pickle.csv')

# In[13]:

def CompereAlgo(testTrainCV = 2 ,trainValCV = 2,randomSearchIter = 2,debug = False):
    """main function to this module, get parmeters for training and create the mesures with all 
       the models 
        
    
        ----------
        testTrainCV,trainValCV : {int}  CV amount for test-train and train-validation
        
        randomSearchIter : {int}  iteration number for random search for each model
        
        Returns
        -------

            
    """
    global INSTANCES_TO_MEASURE_TIME,TEST_TRAIN_CV,TRAIN_VALIDATION_CV,RANDOM_SEARCH_ITER
    ##global
    TEST_TRAIN_CV = testTrainCV
    TRAIN_VALIDATION_CV = trainValCV
    RANDOM_SEARCH_ITER = randomSearchIter
    
    allDataSetsPaths = getsDataPaths()
    models = CreateModels()
    RunModels(models,allDataSetsPaths)
    
    ##debug mode
    if (debug):
        for i,path in enumerate(allDataSetsPaths): 
            name = path.split('\\')[-1].split('.')[0]
            X, y = preprocess(path)
            print ('id', i ,name, 'shape: ', X.shape, 'with', len(y.unique()),'labels') 
    

# CompereAlgo()

def setClassColumn(data,names):
    columns = data.columns
    indexOfClass = np.where(columns.str.lower().isin(names) == True)
    if len(indexOfClass[0]) == 0:
        return
    columns = list(columns)
    pos1 = indexOfClass[0][0]
    pos2 = len(columns)-1
    data[columns[pos1]], data[columns[pos2]] = data[columns[pos2]], data[columns[pos1]] 
    columns[pos1], columns[pos2] = columns[pos2], columns[pos1] 
    data.columns = columns
    
test = pd.read_csv(getsDataPaths()[115])


setClassColumn(test,["class"])




