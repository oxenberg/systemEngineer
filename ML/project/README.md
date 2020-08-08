# Final Project 

This Final Project in advanced ML course compares 4 Tree Based Ensemble models. The Goal of the project is to compare the performance of un-familiar models (models that weren't recognised like XGBoost or Random-Forest), to a well-known model. 
In this project we compare the 3 folowing models: InfiniteBoost, KTboost, and NGboost to LogitBoost. 

### Models
#### InfiniteBoost
InfiniteBoost is an approach to building ensembles which combines best sides of random forest and gradient boosting. 

[Git](https://github.com/arogozhnikov/infiniteboost)

[Paper](https://arxiv.org/abs/1706.01109)

Hyperparameters - 
1. max_depth: default = 3
2. max_features: default = 1
3. learning_rate: default = 0.1
4. use_all_in_update: if true, all the data are used in setting new leaves values
5. param loss: any descendant of AbstractLossFunction, those are very various.default = None
6. n_estimators: number of trained trees in the ensemble. default = 10
7. subsample: fraction of data to use on each stage of boosting, or "bagging" for bagging strategy (with replacement)

#### KTBoost
KTBoost implements several boosting algorithms with different combinations of base learners, optimization algorithms, and loss functions.

[Git](https://github.com/fabsig/KTBoost)

[Paper](https://arxiv.org/abs/1902.03999)

Hyperparameters - 
1. loss : loss function to be optimized.
2. update_step : string, default="hybrid". Defines how boosting updates are calculated. 
3. base_learner : string, default="tree". Base learners used in boosting updates. 
4. learning_rate : float, optional (default=0.1)
5. n_estimators : int (default=100)
6. max_depth : integer, optional (default=5)
7. min_samples_leaf : int, float, optional (default=1)
8. criterion : string, optional (default="mse"). The function to measure the quality of a split.
9. kernel : string, default="rbf". Kernel function used for kernel boosting. 
10. theta : float, default: 1. Range parameter of the kernel functions which determines how fast the kernel function decays with distance.
11. n_neighbors : int, default: None. If the range parameter 'theta' is not given, it can be determined from the data using this parameter. 
12. alphaReg : float, default: 1. Regularization parameter for kernel Ridge regression boosting updates.
13. nystroem : boolean, default=None. Indicates whether Nystroem sampling is used or not for kernel boosting.
14. n_components : int, detault = 100. Number of data points used in Nystroem sampling for kernel boosting.

#### NGBoost
NGBoost is a Python library that implements Natural Gradient Boosting, as described in "NGBoost: Natural Gradient Boosting for Probabilistic Prediction". 

[Git](https://stanfordmlgroup.github.io/projects/ngboost/)

[Paper](https://arxiv.org/abs/1910.03225)

Hyperparameters - 
1. Dist: This parameter sets the distribution of the output. Currently, the library supports Normal, LogNormal, and Exponential distributions for regression, k_categorical and Bernoulli for classification. Default: Normal
2. Score: This specifies the scoring rule. Currently, the options are between LogScore or CRPScore. Default: LogScore
3. Base: This specifies the base learner. This can be any Sci-kit Learn estimator. Default is a 3-depth Decision Tree
4. n_estimators: The number of boosting iterations. Default: 500
5. learning_rate: The learning rate. Default:0.01
6. minibatch_frac: The percent subsample of rows to use in each boosting iteration. This is more of a performance hack than performance tuning. When the data set is huge, this parameter can considerably speed things up.

### Quick Start
In order to run the code there a few neccesary steps: 

1. Clone This repository

2. Clone infinteboost repo using the following command. Make sure it's in the project folder. 

`$ git clone [InfinteBoost](https://github.com/arogozhnikov/infiniteboost.git)` 

3. Install the required packages (according to the requirments file)

`$ pip install -r requirments.txt` 

4. Create a data folder and put all datasets in the folder. The folder should be at the root directory of this git repository. 

5. run main.py from code folder

### Documentation
#### Python Files 

1. CompareAlgo - The file contains the code for the part C of the project - examining and comparing all algorithms on the 150 classification datasets. This part includes the preprocessing we need to do for the datasets (complete missing values, encode label columns and deal with problametic labels), running all the algorithms on all datasets using RandomizedSearchCV and saving all measures to file. 

2. Statistics - The File contains the code for part D of the project - the Friedman's Test and the post-hoc tests we did on the data. 

3. MetaClassifier - The file contains the code for Part E of the project - extracting the leading algorithm for each dataset, building and running the MetaClassifier and extracting the importance measures. 

4. main - The main file of the project - responsible of connecting and running all parts of the project. 

5. restoreData - The file will help us restore the data in case we had any errors and our data is distributed between multiple files. Full instruction on how to use the code are presented in the file itself. 

6. infoGraphic - The file contains the code for all the graphs we added in the conclusions part in the report. 

#### Errors 

Collecting the measures for the first part of the project is done while appending the data to a file. During our run we encountered errors with some of the datasets, and had to resume the code to collect the rest of the data. When it happened, we saved the data collected so far in the folder data/backup and changed the name to "measuresPart{i}", where i is the number of the file. 
In case of errors with running the first part of the project (in the compareAlgo part), you should move the data file to the data/backup folder, and change the file name. After completing the entire run of the model in the first part, you will need to run the restoreData file with the instructions there. After completing this procedure, you will be able to run the rest of the project. 

We will mention that there are no errors in the code that we know of. The code crushed for us in the last algorithm, in the 10 cv in one of the datasets - so we resumed the run from this point forward. 
