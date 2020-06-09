# HW-3 image classification

## info
-------------
we used 2 model:
- **model_1**: imagenet/nasnet_large/classification, [link] (https://tfhub.dev/google/imagenet/nasnet_large/classification/4)
- **model_2**: imagenet/mobilenet_v2_100_224/feature_vector, [link] (https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4)


## results
-------------
model  |accuracy | loss | epcoh |
------ | ------- | -----|-------|
1      | 0.8259  |1.6051| 5     |
2      | 0.9410  |1.2423| 5     |


### grapths
-------------

#### model1
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/acc.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_1/loss.png)

#### model2
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/acc.png)
![](https://github.com/oxenberg/systemEngineer/blob/master/ML/HW_3/graphs/model_2/loss.png)

