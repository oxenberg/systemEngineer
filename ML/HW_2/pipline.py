## all imports 
import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Dropout
from mlxtend.classifier import EnsembleVoteClassifier,StackingCVClassifier

from kaggle.api.kaggle_api_extended import KaggleApi
API = KaggleApi({"username":"oroxenberg","key":"538527cfd22bdb19ae607236e4b44cc8"})
API.authenticate()


def downloadAndOpen():
    import pathlib
    import zipfile
    API.competition_download_files('bgutreatmentoutcome')
    with zipfile.ZipFile(pathlib.Path().absolute()/"bgutreatmentoutcome.zip", 'r') as zip_ref:
        zip_ref.extractall(pathlib.Path().absolute())
    



    
def checkScore(clf,ID, test_processed):
    prob = clf.predict_proba(test_processed)
    df2 = pd.DataFrame({"id" : [x for x in range(1, len(prob)+1)],"ProbToYes" : list(prob[:,1])})
    df2.to_csv(f'submission{ID}.csv', index = None)
    df2.to_csv('submission.csv', index = None)
    API.competition_submit('submission.csv','API Submission','bgutreatmentoutcome')
    
    
def testModel(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(model.score(X_test,y_test))

    

def preprocess(train,test):
    #concate data
    train['dataType'] = 'train'
    test['dataType'] = 'test'
    test["CLASS"] = ""
    labelColumns = ["CLASS","dataType"]
    data = pd.concat([train, test])
    
    #handle missing value
    n = data.notnull()
    precentNotNull = 0.8
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


def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature



# --------- load the data ---------
missing_values = ["n/a", "Nane", "nan","Null"]

train = pd.read_csv('./data/train.CSV', na_values = missing_values)
test = pd.read_csv('./data/test.CSV', na_values = missing_values)

train,test = preprocess(train,test)
train['CLASS'] = train['CLASS'] =='Yes'

# #handle inbulnce
# classCounter = train["CLASS"].value_counts()
# numberToAdd = abs(classCounter["Yes"] - classCounter["No"])
# train = train.append(train[train["CLASS"] == classCounter.index[classCounter.argmin()]].sample(n=numberToAdd, random_state=1),ignore_index = True)

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

num_feats =  150

### Feature selection
## Pearson correlation
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

## chi-square

X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

## recursive feature elimination
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

## Lasso 
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

## tree based - select from model
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')

##light gbm - select from model
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
embeded_lgb_selector.fit(X, y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

## final step
feature_selection_df = pd.DataFrame({'Feature':X.columns, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)

selected_features = feature_selection_df[feature_selection_df['Total']>=3]

selected_features = list(selected_features['Feature'])

test = test[selected_features]
X = X[selected_features]
Y = y




def build_model(nodes1=10,nodes2=10,nodes3=10, lr=0.001,lyers1=1,lyers2=1,lyers3=1,activation = 'relu', input_shape=X.shape[1]):
    model = Sequential()
    model.add(Dense(nodes1,activation=activation, kernel_initializer='uniform', input_shape=(input_shape,)))
    for l in range(lyers1):
        model.add(Dense(nodes1,activation=activation))
    for l in range(lyers2):
        model.add(Dense(nodes2,activation=activation))
    for l in range(lyers3):
        model.add(Dense(nodes3,activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.SGD(lr=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return(model)



#--------- model ---------
#----random search-----

#- Boosted tree -
# gbc = GradientBoostingClassifier(verbose = 1)
# distributions = {'n_estimators':[50, 100, 200], 'max_depth': list(np.arange(3, 10)), 'min_samples_split':list(np.arange(2, 10))}
# clf = RandomizedSearchCV(gbc, distributions,n_iter = 10, random_state=42,verbose = 10, cv=3)
# fit = clf.fit(train_processed.iloc[:, :-1], train_processed.iloc[:,-1])
# sorted(fit.cv_results_.keys())
# fit.cv_results_
# fit.best_params_

# -Adaboost-
# ada = AdaBoostClassifier()
# distributions = {'n_estimators':[500, 700, 1000], 'learning_rate': list(np.arange(0.05, 0.5, 0.05))}
# clf = RandomizedSearchCV(ada, distributions,n_iter = 5, random_state=42,verbose = 10, cv=3)
# fit = clf.fit(train_processed.iloc[:, :-1], train_processed.iloc[:,-1])
# sorted(fit.cv_results_.keys())
# fit.cv_results_
# fit.best_params_

# -XGboost-
# xg = xgb.XGBClassifier(verbosity = 1)
# distributions = {'n_estimators':[40, 100, 200], 'booster': ['gbtree', 'dart'], 'eta': list(np.arange(0.1, 1, 0.1)),
#                  'gamma' : [0, 0.1], 'max_depth' : [6, 10, 15]}
# clf = RandomizedSearchCV(xg, distributions,n_iter = 10, random_state=42, verbose = 10, cv=5)
# fit = clf.fit(train_processed.iloc[:, :-1], train_processed.iloc[:,-1])
# sorted(fit.cv_results_.keys())
# fit.cv_results_
# fit.best_params_

# # -Logistic Regression-
# logr = LogisticRegression(verbose = 1)
# distributions = {'penalty':['l1', 'l2'], 'C': list(np.arange(0.1, 1, 0.1))}
# clf = RandomizedSearchCV(logr, distributions,n_iter = 10, random_state=42, verbose = 10, cv=5)
# fit = clf.fit(train_processed.iloc[:, :-1], train_processed.iloc[:,-1])
# sorted(fit.cv_results_.keys())
# fit.cv_results_
# fit.best_params_

# -Random Forest-
# rf = RandomForestClassifier(verbose = 1)
# distributions = {'n_estimators':[40, 100, 200], 'criterion': ['gini', 'enthropy'], 'min_samples_split':list(np.arange(2, 10))}
# clf = RandomizedSearchCV(rf, distributions,n_iter = 10, random_state=42, verbose = 10, cv=5)
# fit = clf.fit(train_processed.iloc[:, :-1], train_processed.iloc[:,-1])
# sorted(fit.cv_results_.keys())
# fit.cv_results_
# fit.best_params_

#-NN-
# nodes1 = [500,700,1000]
# nodes2 = [300,400,500]
# nodes3 = [100,200,300]# number of nodes in the hidden layer
# lrs = [0.001, 0.002, 0.003] # learning rate, default = 0.001
# Lyers1 = [1,2,3]
# Lyers2 = [1,2,3]
# Lyers3 = [1,2]

# epochs = 10
# batch_size = 32

# model = KerasClassifier(build_fn=build_model, epochs=epochs,
#                         batch_size=batch_size, verbose=1)
# param_distributions = dict(nodes1=nodes1,nodes2=nodes2,nodes3=nodes3,
#                            lr=lrs, lyers1 = Lyers1, lyers2 = Lyers2, lyers3 = Lyers3)

# grid = RandomizedSearchCV(estimator=model,n_iter = 30, param_distributions =param_distributions, cv=5,
#                     n_jobs=1, verbose=1)

# clf = grid.fit(X, Y)

# -SVM-

# svm = SVC(gamma='auto', probability = True)
# distributions = {'C':[0.1, 0.5, 1], 'kernel': ['linear', 'rbf']}
# clf = RandomizedSearchCV(svm, distributions,n_iter = 5, random_state=42, verbose = 10, cv=5)
# clf.fit(X,Y)

# -KNN-
# neigh  = KNeighborsClassifier()
# distributions = {'n_neighbors':np.arange(1,10,1)}
# clf = BayesSearchCV(neigh , distributions,n_iter = 5, random_state=42, verbose = 10, cv=5)
# clf.fit(X,Y)


#  # -voting classifier-
nodes1 = [500,700,1000]
nodes2 = [300,400,500]
nodes3 = [100,200,300]# number of nodes in the hidden layer
lrs = [0.001, 0.002, 0.003] # learning rate, default = 0.001
Lyers1 = [1,2,3]
Lyers2 = [1,2,3]
Lyers3 = [1,2]
# activationfuncs = ['relu']

epochs = 10
batch_size = 32

clf1 = LogisticRegression()
clf2 = xgb.XGBClassifier()
clf3 = AdaBoostClassifier()
clf4 = GradientBoostingClassifier()
clf5 = RandomForestClassifier()
clf6 = KerasClassifier(build_fn=build_model, epochs = epochs, batch_size = batch_size)
clf6._estimator_type = "Classifier"
clf7 =  GaussianNB()

nodes1_lr = [10,20,30]
nodes2_lr = [10,20,30]
lrs_lr = [0.001, 0.002, 0.003] # learning rate, default = 0.001
Lyers1_lr = [1,2,3]
Lyers2_lr = [1,2,3]
Lyers3_lr = 0
# activationfuncs_lr = 'relu']
epochs_lr = 8
batch_size_lr = 16

lr = KerasClassifier(build_fn=build_model, epochs = epochs_lr, batch_size = batch_size_lr, lyers3 = Lyers3_lr, input_shape = 14)
lr._estimator_type = "Classifier"

svc = StackingCVClassifier(classifiers=[clf1,clf2,clf3,clf4,clf5,clf6, clf7],meta_classifier=lr,random_state=42,use_probas=True, verbose = 1)


distributions = { 'logisticregression__C': list(np.arange(0.1,1)), 'xgbclassifier__n_estimators': [40,100, 200], 
                  'xgbclassifier__booster': ['gbtree', 'dart'], 'xgbclassifier__eta': list(np.arange(0.1, 1, 0.1)),'xgbclassifier__gamma' : [0, 0.1], 
                  'xgbclassifier__max_depth' : [6, 10, 15], 'adaboostclassifier__n_estimators':[40, 50, 60], 'adaboostclassifier__learning_rate': list(np.arange(0.05, 0.5, 0.05)), 
                 'gradientboostingclassifier__n_estimators':[50, 100, 200], 'gradientboostingclassifier__max_depth': list(np.arange(3, 10)), 'gradientboostingclassifier__min_samples_split':list(np.arange(2, 10)), 
                 'randomforestclassifier__n_estimators':[40, 100, 200],'randomforestclassifier__min_samples_split':list(np.arange(2, 10)), 
                 'kerasclassifier__nodes1': nodes1,'kerasclassifier__nodes2': nodes2,'kerasclassifier__nodes3': nodes3, 'kerasclassifier__lr':lrs , 'kerasclassifier__lyers1' : Lyers1,
                 'kerasclassifier__lyers2' : Lyers2,'kerasclassifier__lyers3' : Lyers3 , 
                'meta_classifier__nodes1' : nodes1_lr,'meta_classifier__nodes2' : nodes2_lr,'meta_classifier__lyers1' : Lyers1_lr,'meta_classifier__lyers2' : Lyers2_lr,
                  'meta_classifier__lr' : lrs_lr }

# 'kerasclassifier__activation' : activationfuncs,'meta_classifier__activation' : activationfuncs_lr,'meta_classifier__lyers2' : Lyers2_lr,


clf = BayesSearchCV(svc, distributions,n_iter = 1, random_state=42, verbose = 10, cv=2)
fit = clf.fit(X, Y)
sorted(fit.cv_results_.keys())
fit.cv_results_
fit.best_params_


# -autoML- 

#clf = autosklearn.classification.AutoSklearnClassifier()
#clf.fit(X, Y)

#---------test the model---------


testModel(clf, X, Y)


#---------submit---------

checkScore(clf,40, test)

bestVals  = dict(clf.best_params_)

pd.DataFrame(bestVals, index=[1,]).to_csv("log.csv",mode='a',header=False)

















