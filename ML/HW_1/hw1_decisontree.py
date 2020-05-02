# -*- coding: utf-8 -*-
"""HW1_DecisonTree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TIghJMqc6hK0dtubR3Wl-kGrfypTd7m7
"""

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import random

"""# Build tree"""

class SplitFactor():
  def __init__(self,col,val,featureType,optionsDict):
    self.col = col
    self.val = val
    self.featureType = featureType #S - string , N - num 
    self.optionsDict = optionsDict
    if featureType == "S":
         self.optionsDict = {v: k for k, v in self.optionsDict.items()}    
  def calc(self,row):
    if self.featureType == "S":
      try:
          return [1,self.optionsDict[row[self.col]]]
      except KeyError:
          return [0,self.model]
    else: 
      # if val bigger eqqual then the TH opt2 if smaller  to opt1 
      if row[self.col]<self.val:
        return [1,"opt1"]
      else: 
        return [1,"opt2"]
  def setModel(self,model):
      self.model = model

class RegrationTree():

  def __init__(self,trainData,trainLabel,minSample,regModel,StringColumns = []):
    self.regModel = regModel # the user give the Sklearn Model inisitalizer 
    self.minSample = minSample
    self.trainData = trainData
    self.trainLabel = trainLabel
    self.StringColumns = StringColumns
    self.root = {}
    
  @staticmethod
  def MSETest(PredictLabels,labels):
    if isinstance(PredictLabels,pd.Series):
        PredictLabels = PredictLabels.values
    if isinstance(labels,pd.Series):
        labels = labels.values
    mse = 0
    for index in range(len(PredictLabels)):
      mse+=(PredictLabels[index] - labels[index])**2
    mse = mse/len(PredictLabels)
    return mse

  @staticmethod
  def checkIfNum(val):
    try:
        float(val)
        return True
    except:
        return False
    
  @staticmethod
  #return string columns
  def cleanData(data,stringColumns):
    #check float columns
    floatData = data.drop(stringColumns, axis=1)
    data.drop(floatData.columns, axis=1,inplace=True)
    return data , floatData



  def fit(self):
    stringData,_ = self.cleanData(self.trainData.copy(),self.StringColumns)
    self.enc = OneHotEncoder(handle_unknown='ignore')
    self.enc.fit(stringData)
    data = pd.concat([self.trainData, self.trainLabel], axis=1)
    depth = 0
    self.split(self.root,data,depth)
  

  def split(self,node,data,depth):
    depth +=1
    #print(depth)
    data = data.reset_index(drop=True)
    dataString,dataFloat = self.cleanData(data.iloc[:,:-1].copy(),self.StringColumns)
    dataString= pd.DataFrame(self.enc.transform(dataString).toarray(),columns = self.enc.get_feature_names())
    Encode_data = pd.concat([dataString.reset_index(drop=True), dataFloat.reset_index(drop=True),data.iloc[:,-1].reset_index(drop=True)], axis=1)
    model = self.regModel()
    model.fit(Encode_data.iloc[:,:-1],Encode_data.iloc[:,-1])
    if len(data) <= self.minSample or len(data) <= 1:
      node["Regfunc"] = model
      return
    listData = self.MSE_Calc(node,data,Encode_data)
    if listData == None:
      node["Regfunc"] = self.regModel()
      node["Regfunc"].fit(Encode_data.iloc[:,:-1],Encode_data.iloc[:,-1])
      return
    for index,opt in enumerate(node["optionsDict"]):
      node["optionsDict"][opt] = {}
      node["splitFactor"].setModel(model)
      self.split(node["optionsDict"][opt],listData[index],depth)



  # run over all the columns names and calc the max MSE feature and the split factor
  def MSE_Calc(self,node,data,Encode_data):      
    MIN_MSE_ITEM = ["","",np.inf,"",""]
    for col in list(data.columns)[:-1]:
      val = data[col].iloc[0]
      THs = data[col].unique()
      if(self.checkIfNum(val)):
        THs = list(sorted(THs))
        TH_MSE_List = []
        for TH in THs[1:]:
          left = Encode_data[data[col]< TH]
          right = Encode_data[data[col] >= TH]
          optionsList = [left,right]
          MSE = self.findMSE(optionsList)
          TH_MSE_List.append(MSE)
        if len(TH_MSE_List) <=1:
            continue
        minIndex = np.argmin(np.array(TH_MSE_List))
        minVal = min(TH_MSE_List)
        dataForNode =[data[data[col]< THs[minIndex+1]],data[data[col] >= THs[minIndex+1]]]
        currentMSE = [col,THs[minIndex+1],minVal,"N",dataForNode]
      #numinaly
      else:
        optionsList = []
        dataForNode = []
        for TH in THs:
          optionsList.append(Encode_data[data[col] == TH])
          dataForNode.append(data[data[col] == TH])
        if len(dataForNode) <=1:
            continue
        MSE = self.findMSE(optionsList)
        currentMSE = [col,THs,MSE,"S",dataForNode]
      if currentMSE[2] < MIN_MSE_ITEM[2]:
        MIN_MSE_ITEM = currentMSE
    
    
    if MIN_MSE_ITEM[0] == "":
        return None
    optionsDict = {}
    if(MIN_MSE_ITEM[3] == "S" ):
      for index in range(1,len(MIN_MSE_ITEM[1])):
        optionsDict["opt"+str(index)] = MIN_MSE_ITEM[1][index-1]
    else:
      optionsDict["opt1"] = {}
      optionsDict["opt2"] = {}
    SF = SplitFactor(MIN_MSE_ITEM[0],MIN_MSE_ITEM[1],MIN_MSE_ITEM[3],optionsDict)
    node["optionsDict"] = optionsDict.copy()
    node["splitFactor"] = SF
    return MIN_MSE_ITEM[-1]
  
  def findMSE(self,optionsList):
    genralMSE = 0
    for opt in optionsList:
      regModelPersonal = self.regModel()
      x = opt.iloc[:,:-1]
      y = opt.iloc[:,-1]

      regModelPersonal.fit(x,y)
      preY = regModelPersonal.predict(x)
      mse = self.MSETest(preY,y)
          
      genralMSE += mse
    genralMSE = genralMSE/len(optionsList)
    return genralMSE

  def Predict(self,data):
    data = data.reset_index(drop=True)
    PredictValues = []
    cont = False
    dataString,dataFloat = self.cleanData(data.iloc[:,:-1].copy(),self.StringColumns)
    dataString= pd.DataFrame(self.enc.transform(dataString).toarray(),columns = self.enc.get_feature_names())
    Encode_data = pd.concat([dataString.reset_index(drop=True), dataFloat.reset_index(drop=True),data.iloc[:,-1].reset_index(drop=True)], axis=1)
    for index, row in data.iterrows():
      location = self.root
      while ("Regfunc" not in location):
        nextStep = location["splitFactor"].calc(row)
        #we go to the next layer
        if nextStep[0] == 1:
            location = location["optionsDict"][nextStep[1]]
        #we cant go to the next layer and we need to calc the prediction by current regreation
        else:
            model = nextStep[1]
            cont = True
            PredictValues.append(model.predict(Encode_data.iloc[index].values.reshape(1,-1))[0])
            break
      if cont:
        cont = False
        continue
      PredictValues.append(location["Regfunc"].predict(Encode_data.iloc[index].values.reshape(1,-1))[0])
    return PredictValues
  


"""# Main"""


#dataVal = np.array([np.arange(8),np.arange(8)]).T
#DB = pd.DataFrame(dataVal,columns = ["col1","label"])
'''
DB = pd.read_csv("machine.data", delimiter = ",")
DB.columns = ["vendor_name","Model_Name","MYCT","MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"]
'''





def activeCompare(DSname,delimiter,regModel,min_samples_splits,indexsStringLine = [],skiprows = [],rep = None,dropCol = []):
    DB = pd.read_csv("./data/"+DSname, delimiter = delimiter,skiprows=skiprows)

    
    if rep !=None:
        DB = DB.replace(rep, np.nan)
    DB = DB.dropna(axis = 1)
    
    DB.drop(DB.columns[dropCol],axis=1,inplace=True)
    
    colNames = []
    for i in range(DB.shape[1]):
        colNames.append("col"+str(i))
        
    DB.columns = colNames
    
    for index in indexsStringLine:
        DB[colNames[index]] = DB[colNames[index]].apply(lambda x: 'A'+str(x))
    
    
    X_train, X_test, y_train, y_test = train_test_split(DB.iloc[:,:-1], DB.iloc[:,-1], test_size=0.2, random_state=42)
    
    
    dataString,dataFloat = RegrationTree.cleanData(DB.iloc[:,:-1].copy(),DB.columns[indexsStringLine])
    enc = OneHotEncoder(handle_unknown='ignore')
    valuesString = enc.fit_transform(dataString).toarray()
    dataString= pd.DataFrame(valuesString,columns = enc.get_feature_names())
    Encode_data = pd.concat([dataString.reset_index(drop=True), dataFloat.reset_index(drop=True),DB.iloc[:,-1].reset_index(drop=True)], axis=1)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(Encode_data.iloc[:,:-1], Encode_data.iloc[:,-1], test_size=0.2, random_state=42)
    
    print(f"-------{DSname}-------")
    for min_samples_split in min_samples_splits:
        print(f"--Min splits:{min_samples_split}--")
        #sklearn model
        regressor = DecisionTreeRegressor(min_samples_split = min_samples_split)
        regressor.fit(X_train1,y_train1)
        yPred = regressor.predict(X_test1)
        print(f"sklearn model MSE is: {RegrationTree.MSETest(yPred,y_test1.values)}")
#        from sklearn import tree
#        plt.figure(figsize = (10,10))
#        tree.plot_tree(regressor, fontsize = 7) 
        #our model
        
        RT = RegrationTree(X_train,y_train,min_samples_split,regModel,StringColumns = DB.columns[indexsStringLine])
        RT.fit()
        yPred = RT.Predict(X_test)
        print(f"our model MSE is: {RegrationTree.MSETest(yPred,y_test.values)}")

        
        ##random Model
        minVal = np.min(DB.iloc[:,-1].values)
        maxVal = np.max(DB.iloc[:,-1].values)
        yPred = []
        for i in range(len(X_test)):
            yPred.append(random.uniform(minVal, maxVal))
        print(f"Random model MSE is: {RegrationTree.MSETest(yPred,y_test.values)}")

###params for the model####
# you can change the min splits with min_samples_splits and give multiply splits
# you can change the model by import the right model and inisiate regModel var
        
min_samples_splits = [10, 15, 30]
from sklearn.linear_model import Ridge
regModel = Ridge

#params for the preprocess
#---indexsStringLine = index of categorial columns 
#---delimiter
#---dataset name



#working db

activeCompare("machine.data",",",regModel,min_samples_splits,indexsStringLine =  [0,1])

activeCompare("servo.data", ',',regModel,min_samples_splits,indexsStringLine = [0,1])

activeCompare("Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv",';',regModel,min_samples_splits,indexsStringLine = [0])

activeCompare("data_akbilgic.csv",',',regModel,min_samples_splits = min_samples_splits,indexsStringLine = [0])

activeCompare("student-por.csv",';',regModel,min_samples_splits = min_samples_splits,indexsStringLine = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22])


















