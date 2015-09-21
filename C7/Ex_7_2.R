library(mlbench)
library(caret)
library(doMC)

# for parallel
registerDoMC(cores = 4)


set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)

# convert from matrix to data frame
trainingData$x <- data.frame(trainingData$x)

# visulisation
featurePlot(trainingData$x, trainingData$y)

# now use a list with the vector y and matrix of predictors x
# use large test set to etimate error rate with good precison
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)

# tune several models to the data

# set up cross validation method - use bootstrap as only 200 
# samples to train data to
ctrl <- trainControl(method = 'boot632', number = 25)

# start with knn. Remember to center and scale beforehand
# to make sure predictors with larger magnitude do not dominate


set.seed(101)
knnModel <- train(x = trainingData$x,
                  y = trainingData$y,
                  method = 'knn',
                  preProc = c('center', 'scale'),
                  tuneGrid = data.frame(.k = 1:20),
                  trControl = ctrl)

knnPred <- predict(knnModel, newdata = testData$x)

# postResample() gets the test set performance values
postResample(pred = knnPred, obs = testData$y)

# mars
library(earth)
# first get an idea of pruning range

marsFit <- earth(trainingData$x, trainingData$y)
summary(marsFit)
marsFit

# up to 2nd degree, 2:18 
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:18)

# no preprocess needed for mars
set.seed(101)
marsModel <- train(x = trainingData$x,
                  y = trainingData$y,
                  method = 'earth',
                  tuneGrid = marsGrid,
                  trControl = ctrl)

# quick plot to see RMSE for training
xyplot(RMSE ~ nprune,
       data = marsModel$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

# we see 2nd degree model significanly better than 1st

# work out how to tidy up this type of plot
# but we see the fitted splines
# see the effect of 2nd degree in X3
plotmo(marsModel$finalModel)

marsPred <- predict(marsModel, newdata = testData$x)
postResample(pred = marsPred, obs = testData$y)

varImp(marsModel)

# svm (do a radial and poly kernal)
# radial first

# tuneLength uses default grid search of 14 cost values between 2^-2 and 2^11
# sigma is estimated analytically by defaults
# center and scale 
# only tuning over cost (sigma estimated)

svmradialModel <- train(x = trainingData$x,
                        y = trainingData$y,
                        method = "svmRadial",
                        preProc = c("center", "scale"),
                        tuneLength = 10,
                        trControl = ctrl)


svmradialPred <- predict(svmradialModel, newdata = testData$x)
postResample(pred = svmradialPred, obs = testData$y)

# svm poly
# remember more parameters to tune over, build the grid
svmGrid <- expand.grid(degree = 1:3, 
                       scale = c(0.01, 0.005, 0.001), 
                       C = 2^(-2:7))

svmpolyModel <- train(x = trainingData$x,
                        y = trainingData$y,
                        method = "svmPoly",
                        preProc = c("center", "scale"),
                        tuneGrid = svmGrid,
                        trControl = ctrl)

svmpolyPred <- predict(svmpolyModel, newdata = testData$x)
postResample(pred = svmpolyPred, obs = testData$y)

xyplot(RMSE~C | as.factor(scale),
       data = svmpolyModel$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

# if were not just paying here, would carry on playing
# with params until over fitting starts occuring

# neural net
# very sensitive to highly correlated predictors
# so may need to remove if model struggles
findCorrelation(cor(testData$x), cutoff=0.1)
# look ok!

nnGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                      .size = c(1:10),
                      .bag = F)
# refer to notes for formula for MaxNWts. 10 is max number of 
# hidden layers

nnetModel <- train(x = trainingData$x,
                   y = trainingData$y,
                   method = "avNNet",
                   preProc = c("center", "scale"),
                   linout = TRUE,
                   trace = FALSE,
                   MaxNWts = 10 * (ncol(trainingData$x) + 1) + 10 + 1,
                   tuneGrid = nnGrid,
                   trControl = ctrl,
                   maxit = 500)
 
nnetPred <- predict(nnetModel, newdata = testData$x)
postResample(pred = nnetPred, obs = testData$y)

varImp(nnetModel)