# 8.4

# use a single predictor from the solubility data
# and fit several models:

library(caret)
library(rpart)
library(randomForest)
library(Cubist)
library(AppliedPredictiveModeling)
library(doMC)

data(solubility)
registerDoMC(cores = 4)

solTrainMW <- subset(solTrainXtrans, select="MolWeight")
solTestMW <- subset(solTestXtrans, select="MolWeight")

# note the subtlly. Fails when try
# what is going on with the data?
# solTrainMW <- solTrainXtrans$MolWeight
# 316 rows
# but solTrainMW has 951
# huh? test has 316, train has 951. is something mixed up?

# remember we already have train test splits

# regression tree
# tune over depth - keep constant at 1
set.seed(100)
rpartTune <- train(solTrainMW, solTrainY,
                   method = "rpart2",
                   tuneLength = 1)

rpartTest <- data.frame(Method = "RPart",Y=solTestY,
                        X=predict(rpartTune,solTestMW))
# random forest
# this is tuning over mtry
# typically set to n pred / 3
# but as only 1 keep to 1
set.seed(100)
rfTune <- train(solTrainMW, solTrainY,
                method = "rf",
                tuneLength = 1,
                ntree = 1000)

rfTest <- data.frame(Method = "RF",Y=solTestY,
                        X=predict(rfTune,solTestMW))

R2(rfTest$X, rfTest$Y)

# cubist models

cubistTune1.0 <- train(solTrainMW, solTrainY,
                       method = "cubist",
                       verbose = FALSE,
                       metric = "Rsquared",
                       tuneGrid = expand.grid(committees = 1,
                                              neighbors = 0))

cubistTest1.0 <- data.frame(Method = "Cubist1.0",Y=solTestY,
                            X=predict(cubistTune1.0,solTestMW))

cubistTune1.n <- train(solTrainMW, solTrainY,
                       method = "cubist",
                       verbose = FALSE,
                       metric = "Rsquared",
                       tuneGrid = expand.grid(committees = 1,
                                              neighbors = c(3, 5, 7)))

cubistTest1.n <- data.frame(Method = "Cubist1.n",Y=solTestY,
                            X=predict(cubistTune1.n,solTestMW))

cubistTune100.0 <- train(solTrainMW, solTrainY,
                       method = "cubist",
                       verbose = FALSE,
                       metric = "Rsquared",
                       tuneGrid = expand.grid(committees = 100,
                                              neighbors = 0))

cubistTest100.0 <- data.frame(Method = "Cubist100.0",Y=solTestY,
                            X=predict(cubistTune100.0,solTestMW))

cubistTune100.n <- train(solTrainMW, solTrainY,
                         method = "cubist",
                         verbose = FALSE,
                         metric = "Rsquared",
                         tuneGrid = expand.grid(committees = 100,
                                                neighbors = c(3, 5, 7)))

cubistTest100.n <- data.frame(Method = "Cubist100.n",Y=solTestY,
                              X=predict(cubistTune100.n,solTestMW))

# use postResample to get the RMSE and R2 for test
postResample(pred = predict(cubistTune100.n, solTestMW), obs =  solTestY)

# eg
results <- data.frame(rtree = rpartTest$X, rf = rfTest$X)
lapply(results, postResample, obs = solTestY)

# use resamples() to get sample properties for resamples when training
resamps <- resamples(list(c100.n = cubistTune100.n,
               c100.0 = cubistTune100.0 ))

summary(resamps)

