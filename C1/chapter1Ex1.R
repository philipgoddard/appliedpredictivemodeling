library(caret)
library(ggplot2)
library(mlbench)
library(corrplot)
library(e1071)
data(Glass)
str(Glass)

#column 10 is target

#plot matrix of predictors
pairs(Glass[, -10])

#investigate correlation. Some quite high, may consider remove
correlation <- cor(Glass[, -10])
correlation[1:4, 1:4]
corrplot(correlation, order = "hclust")

#skewness
skewValues <- apply(Glass[, -10], 2, skewness)
#K, Ba pretty skewed!

#have a quick look at distributions
qplot(Glass$RI, geom='density')
qplot(Glass$Na, geom='density')

#look for outliers, boxplots can be handy
boxplot(Glass$RI)

#to preprocess, center, scale, box-cox for skew, and spatialsign for variables
#with significant outliers could be helpful
# REMEMBER tree based models not particulary sensitive to outliers