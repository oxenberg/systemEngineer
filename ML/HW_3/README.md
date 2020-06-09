# HW-3 image classification

## info
-------------
we used 2 model:
- **model_1**: imagenet/nasnet_large/classification, [link] (https://tfhub.dev/google/imagenet/nasnet_large/classification/4)
- **model_2**: imagenet/mobilenet_v2_100_224/feature_vector, [link] (https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4)
			 : [article](https://arxiv.org/pdf/1801.04381.pdf)


## results
-------------
model  |train/val/test |accuracy -test | accuracy -val |accuracy -train| loss - test | epcoh |
------ |-------------- | ------------- |-------------- |-------------- | ------------|-------|
1      |0.7/0.15/0.15  |     0.8726    |     0.8553    |     0.8286    |   1.4240    |   8   |
1      |0.3/0.3/0.4    |     0.7771    |     0.8080    |     0.7204    |   1.7082    |   8   |
2      |0.7/0.15/0.15  |     0.9410    |     0.9457    |     0.8909    |   1.2423    |   5   |
2      |0.3/0.3/0.4    |     0.8664    |     0.8742    |     0.8026    |   1.5016    |   5   |


### grapths
-------------

#### model1
##### 0.7/0.15/0.15
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/acc.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/loss.png)

##### 0.3/0.3/0.4
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/acc2.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/loss2.png)

#### model2
##### 0.7/0.15/0.15
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/acc.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/loss.png)

##### 0.3/0.3/0.4
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/acc2.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/loss2.png)
