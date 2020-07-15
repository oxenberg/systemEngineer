#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt

import numpy as np
import joblib
import pandas as pd
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,precision_score,average_precision_score
from sklearn.datasets import fetch_covtype, load_svmlight_file
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge

#ktboost model
import KTBoost.KTBoost as KTBoost


from tqdm import tqdm


import datetime
import sys
import os

import pathlib

from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from logitboost import LogitBoost


# In[5]:


sys.path.append("../infiniteboost/research")
from SparseInfiniteBoosting import InfiniteBoosting



# In[7]:


sys.path.append("../AdaFair/")
from AdaFair import AdaFair


# In[8]:


allDataSetsPaths = []
dataSetName = "classification_datasets"
for file in os.listdir(f"../{dataSetName}"):
    if file.endswith(".csv"):
        allDataSetsPaths.append(os.path.join(f"../{dataSetName}", file))


# In[9]:


def getBadLabel(data):
    TH_to_other = 10
    countSeries = data.iloc[:,-1].value_counts()
    badLabels = countSeries[countSeries< TH_to_other].index
    return badLabels

def getDataFromPath(path):
    data = pd.read_csv(path)
    data = data.dropna()
    badLabels = getBadLabel(data)
    data.iloc[:,-1] = data.iloc[:,-1].apply(lambda x : "ohter" if x in list(badLabels) else x)

    #check if we still have bad lables under the TH
    badLabels = getBadLabel(data)
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


# In[10]:


for i,path in enumerate(allDataSetsPaths): 
    name = path.split('\\')[-1].split('.')[0]
    X, y = getDataFromPath(path)
    # print ('id', i ,name, 'shape: ', X.shape, 'with', len(y.unique()),'labels') 


# ## exemple

# In[11]:


np.random.seed(42)
models = {}

distributions = dict(estimator__n_estimators=np.arange(50,300,20),estimator__max_leaf_nodes = [2,4,5])
models['infiboost'] = [InfiniteBoosting(),distributions]

distributions = dict(estimator__max_depth=np.arange(3,10),estimator__n_estimators=np.arange(50,300,20))
models['KTBoost'] = [KTBoost.BoostingClassifier(),distributions]

distributions = dict(estimator__n_estimators = np.arange(50,300,20),estimator__Base__max_depth=np.arange(4,10,1))
models["NGBClassifier"] = [NGBClassifier(Dist=Bernoulli),distributions]

distributions = dict(estimator__n_estimators = np.arange(50,300,20),estimator__learning_rate=np.arange(0.1,1.5,0.1))
models["LogitBoost_baseClassfier"] = [LogitBoost(),distributions]

# In[12]:




measuers = pd.DataFrame(columns = ["Dataset_Name","AlgoName","CrossVal","HP_vals",
                                   "ACC","TPR","FPR","Precsion","ROC","Precstion_Recall",
                                   "Training_Time","Inference_Time"])

index = 0
for AlgoName,items in models.items():
    model = items[0]
    dist = items[1]
    print(f"run on {AlgoName}..")
    for datasetName in tqdm(allDataSetsPaths[:3]):
        X, y = getDataFromPath(datasetName)
        X = X.values
        y = y.values
        multiclass = False
        if(len(np.unique(y))>2):
                multiclass = True
        folder = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cvIndex = 1
        for train_indices, test_indices in folder.split(X, y):
            print(f"cv num {AlgoName}..")
            row = []
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
 
            clf = OneVsRestClassifier(clone(model))

            RS = RandomizedSearchCV(clf, dist, random_state=42,n_iter = 2,cv = 2)

            now = datetime.datetime.now()
            RS.fit(X_train, y_train)
            time_stop = datetime.datetime.now()

            y_pred = RS.predict(X_test)
            y_pred_proba = RS.predict_proba(X_test)

            #collect measures
            Dataname = datasetName.split("\\")[1].split(".")[0]
            row.append(Dataname)
            row.append(AlgoName)
            row.append(cvIndex)
            row.append(str(RS.best_params_))
            acc = accuracy_score(y_test,y_pred)
            row.append(acc)

        #     #soultion for multiclass
        #     for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

        #     fpr, tpr, _ = roc_curve(y_test,y_pred)
            row.append("tpr")
            row.append("fpr")
            precsion = precision_score(y_test, y_pred, average='macro')
            row.append(precsion)
            if(multiclass):
                ROC = roc_auc_score(y_test, y_pred_proba,multi_class = "ovr")
            else:
                ROC = roc_auc_score(y_test, y_pred_proba[:,1])
            row.append(ROC)
        #     APS = average_precision_score(y_test,y_pred_proba)
            row.append("APS")
            T_time = str(time_stop- now)
            row.append(T_time)
            row.append("Inference_Time")

            measuers.loc[index] = row
            cvIndex+=1
            index+=1
    #     del clf




