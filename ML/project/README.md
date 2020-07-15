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

### Quick Start
In order to run the code there a few neccesary steps: 

1. Clone This repository

2. Clone infinteboost repo using the following command. Make sure it's in the project folder. 

`$ git clone [InfinteBoost](https://github.com/arogozhnikov/infiniteboost.git)` 

3. Install the required packages (according to the requirments file)

`$ pip install -r requirments.txt` 

4. run main.py from code folder
