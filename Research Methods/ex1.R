data = read_csv('../Desktop/Masters/courses/Research Methods/ex/targil1.csv')


### Question 2.1
# statistics summary
summary(data)
# variance and std
var(data$NumFeatures)
sd(data$NumFeatures)
sd(data$`ACC-RF`)
sd(data$`ACC-ANN`)
# skewness and kurtosis
skewness(data$NumFeatures, na.rm = TRUE)
skewness(data$`ACC-RF`, na.rm = TRUE)
skewness(data$`ACC-ANN`, na.rm = TRUE)
kurtosis(data$NumFeatures, na.rm = TRUE)
kurtosis(data$`ACC-RF`, na.rm = TRUE)
kurtosis(data$`ACC-ANN`, na.rm = TRUE)

## plot 
ggplot(data, aes(x = data$NumFeatures))+geom_histogram()+xlab('Num Features')
ggplot(data, aes(x = data$`ACC-RF`))+geom_histogram()+xlab('ACC - RF')+ylim(0,1.1)
ggplot(data, aes(x = data$`ACC-ANN`))+geom_histogram()+xlab('ACC - ANN')+ylim(0,4.2)

hist(data$NumFeatures, breaks=10, col="blue", xlab = 'num features')

### Question 2.2
## corelations 
# pearson correlation
cor(data$NumFeatures,data$`ACC-RF`, use="na.or.complete", method = "pearson")
cor(data$NumFeatures, data$`ACC-ANN`, use="na.or.complete", method = "pearson")
cor(data$`ACC-RF`, data$`ACC-ANN`, use="na.or.complete", method = "pearson")
# kendal correlation
cor(data$NumFeatures, data$`ACC-RF`, use="na.or.complete", method = "kendal")
cor(data$NumFeatures, data$`ACC-ANN`, use="na.or.complete", method = "kendal")
cor(data$`ACC-RF`, data$`ACC-ANN`, use="na.or.complete", method = "kendal")

### Question 2.3
## variance test

var.test(data$`ACC-RF`,data$`ACC-ANN`) # the variance are the same
## t-test

t.test(data$`ACC-RF`,data$`ACC-ANN`, var.equal = TRUE)

### Question 2.4 
# not sure if it should beb paired t-test or not
t.test(data$`ACC-RF`[data$NumFeatures>100],data$`ACC-RF`[data$NumFeatures<=100], paired = TRUE)
t.test(data$`ACC-ANN`[data$NumFeatures>100],data$`ACC-ANN`[data$NumFeatures<=100], paired = TRUE)

### Question 2.5
t.test(data$`ACC-RF`, mu=85.6 , alt="two.sided", conf.level = 0.95)
t.test(data$`ACC-ANN`,mu=85.6 , alt="two.sided", conf.level = 0.95)




