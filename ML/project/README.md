# Final Project 

This Final Project in advanced ML course compares 4 Tree Based Ensemble models. The Goal of the project is to compare the performance of un-familiar models (models that weren't recognised like XGBoost or Random-Forest), to a well-known model. 
In this project we compare the 3 folowing models: InfiniteBoost, KTboost, and NGboost to LogitBoost. 

### Models
#### InfiniteBoost
InfiniteBoost is an approach to building ensembles which combines best sides of random forest and gradient boosting. 

[Git](https://github.com/arogozhnikov/infiniteboost)

[Paper](https://arxiv.org/abs/1706.01109)

#### KTBoost
KTBoost implements several boosting algorithms with different combinations of base learners, optimization algorithms, and loss functions.

[Git](https://github.com/fabsig/KTBoost)

[Paper](https://arxiv.org/abs/1902.03999)

#### NGBoost
NGBoost is a Python library that implements Natural Gradient Boosting, as described in "NGBoost: Natural Gradient Boosting for Probabilistic Prediction". 

[Git](https://stanfordmlgroup.github.io/projects/ngboost/)

[Paper](https://arxiv.org/abs/1910.03225)

Hyperparameters - 

### Quick Start
In order to run the code there a few neccesary steps: 

1. Clone This repository

2. Clone infinteboost repo using the following command. Make sure it's in the project folder. 

`$ git clone [InfinteBoost](https://github.com/arogozhnikov/infiniteboost.git)` 

3. Install the required packages (according to the requirments file)

`$ pip install -r requirments.txt` 

4. run main.py from code folder

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

We will mention That there are no errors in the code that we know of. The code crushed for us in the last algorithm, in the 10 cv in one of the datasets - so we resumed the run from this point forward. 
