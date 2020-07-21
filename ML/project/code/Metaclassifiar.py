# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:57:57 2020

@author: oxenb
"""
import pandas as pd 
import json
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score




###global



##read mesure

def TranformReslut(measuers_input,metaFeatures):
    """calculate the wining algo for each dataset from measuers_input table and add 2 columns to metaFeatures table.
       The first indicate the model used and the second column indicate if he got the best score with bool value. 
        
        ----------
        measuers_input : {dataFrame}  measures from the CompereAlgo moudle
        
        metaFeatures : {dataFrame}  meta features table 
        
        Returns
        -------

            
    """
    #create table of who won in each dataset
    winAlgo= measuers_input.iloc[measuers_input.groupby(['Dataset_Name'])['AUC'].idxmax()][["Dataset_Name","AlgoName"]]
    winAlgo["win"] = True
    scores = measuers_input[["Dataset_Name","AlgoName"]].drop_duplicates()
    winAlgo = pd.merge(winAlgo, scores, on = ["Dataset_Name","AlgoName"], how = 'outer').fillna(False)
    
    #filtter datasets that are not in metaFeatures table
    metaFeatures = metaFeatures[metaFeatures['Dataset_Name'].isin(  winAlgo['Dataset_Name'].unique())]
   
    #create the metadata table, combination of metaFeatures with the win algo as binary
    metadata = pd.merge(winAlgo, metaFeatures, on='Dataset_Name', how='left')
    
    #check mismatched after merge
    if metadata.isnull().values.any():
        raise Exception("Sorry... need to fix mismatched columns")
    
    return metadata




def ReadResults():
    """read results from CompereAlgo moudle and change dataset name column name to 'Dataset_Name'.
       read meta features table
       Fix mismatched columns bettwen meta features table and measuers_input.
        
        ----------
        measuers_input : {dataFrame}  measures from the CompereAlgo moudle
        
        metaFeatures : {dataFrame}  meta features table 
        
        Returns
        -------

            
    """
    #read measuers_input
    measuers_input = pd.read_pickle('../data/results/measuers_pickle.csv')
    measuers_input["HP_vals"] = measuers_input["HP_vals"].apply(lambda x : json.loads(x.replace("\'", "\"")))
    
    #read metaFeatures
    metaFeatures = pd.read_csv('../data/ClassificationAllMetaFeatures.csv')
    
    #change dataset column name to be the same
    columns = list(metaFeatures.columns)
    columns[0] = 'Dataset_Name'
    metaFeatures.columns = columns
    #fix diff in dataset names on both tables
    metaFeatures.replace("abalone","abalon",inplace = True)
    

    
    return TranformReslut(measuers_input,metaFeatures)
    




##preprocess
def preprocess(metadata):
    """clean null values and columns with same values, activate one hot encoder on the algo name column
        
        ----------
        metadata : {dataFrame} 
                
        Returns
        -------
        metadata : {dataFrame} 
            
    """
    #remove all nan columns
    metadata = metadata.dropna(how = 'all', axis = 1)
    #remove all same columns
    nunique = metadata.apply(pd.Series.nunique)
    colsToDrop = nunique[nunique == 1].index
    metadata = metadata.drop(colsToDrop, axis=1)
    
    df_ohe_features = pd.get_dummies(metadata["AlgoName"],prefix = "AlgoName" ,columns = "AlgoName" )
    
    metadata = pd.concat([metadata,df_ohe_features],axis =1 )
    metadata.drop("AlgoName", inplace=True,axis = 1)
    return metadata




def calcMeasures(metadata,xgb_model):   
    """calculate the xgboost model over all the datasets with leave on out and get the measures and export to csv
        
        ----------
        metadata : {dataFrame} 
        xgb_model : {xgb sklearn}          
        Returns
        -------
        metadata : {dataFrame} 
        
    """
    allDatasets = metadata["Dataset_Name"].unique()
    measuers = pd.DataFrame(columns = ["dataset_name","ACC","AUC","importance_cover","importance_gain","importance_weight","shap"])
    index = 0
    for dataset in allDatasets:
        metadataTest = metadata[ metadata["Dataset_Name"] == dataset]
        metadataTrain = metadata[ metadata["Dataset_Name"] != dataset]
        
        y_train = metadataTrain.pop("win").values
        X_train = metadataTrain.values
        y_test = metadataTest.pop("win").values
        X_test = metadataTest.values
        
        X_train =X_train[:,1:]
        X_test = X_test[:,1:]
        
        xgb_model.fit(X_train,y_train)
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        
        ACC = accuracy_score(y_test,y_pred)
        AUC = roc_auc_score(y_test, y_pred_proba[:,1], average = 'macro')
        importance_cover = xgb_model.get_booster().get_score(importance_type= "cover")
        importance_gain = xgb_model.get_booster().get_score(importance_type= "gain")
        importance_weight = xgb_model.get_booster().get_score(importance_type= "weight")
        booster = xgb_model.get_booster()
        shapVal = booster.predict(xgb.DMatrix(X_test), pred_contribs=True)
        measuers.loc[index] = [dataset,ACC,AUC,importance_cover,importance_gain,importance_weight,shapVal]
        index +=1


        measuers.to_csv("../data/results/measuers_meta.csv")
        measuers.to_pickle('../data/results/measuers_meta_pickle.csv')
        


def runMetaclassifier():
    """main for this moudle
        
        ----------
                
        Returns
        -------
            
    """
    metadata = ReadResults() 

    metadata = preprocess(metadata)

    ##model
    xgb_model = xgb.XGBClassifier()
    
    calcMeasures(metadata,xgb_model)

