# Q2

# didnt quite understand, so looked at solutions

# we have X1 which is not very granular (2 groups) whic
# is informative for seperating the groups
# and X2 which is very granular, but not informative

set.seed(102)
X1 <- rep(1:2,each=100)
Y <- X1 + rnorm(200,mean=0,sd=4)
set.seed(103)
X2 <- rnorm(200,mean=0,sd=2)
simData <- data.frame(Y=Y,X1=X1,X2=X2)


# look at currentRpart$splits to get the splitting variable
library("rpart")
selectedPredictors <- data.frame(Predictor=as.character())
for (i in 1:100 ) {
  set.seed(i)
  X1 <- rep(1:2,each=100)
  #Y <- X1 + rnorm(200,mean=0,sd=4)
  #Y <- X1 + rnorm(200,mean=0,sd=2)
  Y <- X1 + rnorm(200, mean = 0, sd = 1)
  set.seed(1000+i)
  X2 <- rnorm(200,mean=0,sd=2)
  currentSimData <- data.frame(Y=Y,X1=X1,X2=X2)
  currentRpart <- rpart(Y~X1+X2,data=currentSimData,control=rpart.control(maxdepth=1))
  currentPredictor <- data.frame(Predictor=rownames(currentRpart$splits)[1])
  selectedPredictors <- rbind(selectedPredictors,currentPredictor)
}

table(selectedPredictors)

# we see that even though X2 not informative, as it is more granular
# more similar probability that splits made on it!

# As the amount of noise in the simulation increases, the chances that X2
# are selected increase. Conversely, as the amount of noise decreases the chance that X2 is selected
# decreases. This implies that the granularity provided by X2 has a strong influence on whether or
# not it is selected–not the fact that it has no association with the response.

