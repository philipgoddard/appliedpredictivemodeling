library(caret); library(earth); library(kernlab); library(nnet); library(doMC)
data(tecator)
registerDoMC(cores = 4)

# to do: run, evaluate performance, compare, variable importance, interpret where possible


# absord is a matrix, contain 100 absorbance values for the 215 samples
head(absorp)
summary(absorp)
# endpoints is a matrix containing percentage moisture, fat and protein in
# columns 1-3 respectively
head(endpoints)
summary(endpoints)
# we are predicting upon fat, so this is our target
fat <- endpoints[, 2]

# have a quick look over a few
# play with ggplot, make nicer plots
qplot(absorp[,20], geom='density')

# make data partition
inTrain <- createDataPartition(fat,
                               p=0.75,
                               list = FALSE)

set.seed(100)
# train test split
trainingAbsorp <- absorp[inTrain, ]
testingAbsorp <- absorp[-inTrain, ]
trainingFat <- fat[inTrain]
testingFat <- fat[-inTrain]

# check for near zero variance predictors
NZV <- nearZeroVar(trainingAbsorp, saveMetric = TRUE)
# all predictors are ok in terms of near zero variance

# set up cross validation method - boot632 as training set not that large (only 163 samples)
# dont know why chose 10 fold cv last time...
ctrl <- trainControl(method = 'boot632', number = 25)

# SVM
# remember to center and scale predictors
# (or support vectors will struggle in algorithm)
# no other preprocess needed
# CHECK default value of epsilon
# sigma is estimated, so only tune over cost 
# default is 2^-2 to 2^11 with length 14

set.seed(100)
SVMTune <- train(trainingAbsorp, trainingFat,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 tuneLength = 14,
                 trControl = ctrl)

svmradialPred <- predict(SVMTune, newdata = testingAbsorp)
postResample(pred = svmradialPred, obs = testingFat)

svmResults <- data.frame(fatActual = testingFat, fatPredict = svmradialPred,
                        resid =  svmradialPred - testingFat)

# plots - IN SAMPLE
xyplot(trainingFat ~ predict(SVMTune),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(predict(SVMTune) - trainingFat ~ predict(SVMTune),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(svmResults$fatActual ~ svmResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(svmResults$resid ~ svmResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# check - when doing predictions with this object will it automatically
# center and scale? (the answer must be yes)
# or is it better to explicitly preprocess before feeding
# into the model?

# KNN
# center and scale 
# note if seriously using KNN, would have to remove noisy and
# irrelevant predictors.

tuneGridKNN <- data.frame(k = 1:20)
set.seed(100)
KNNTune <- train(trainingAbsorp, trainingFat,
                 method = "knn",
                 preProc = c("center", "scale"),
                 tuneGrid = tuneGridKNN,
                 trControl = ctrl)

knnPred <- predict(KNNTune, newdata = testingAbsorp)
postResample(pred = knnPred, obs = testingFat)

knnResults <- data.frame(fatActual = testingFat, fatPredict = knnPred,
                         resid =  knnPred - testingFat)


# MARS
# resistant to correlated predictors, transforming and filtering not required
# tune over degree (1:2) and number of terms
set.seed(100)
MARSfit <- earth(trainingAbsorp, trainingFat)
summary(MARSfit)
# get an idea from this that number of terms sensible to tune over
# 2:25 seems reasonable

tuneGridMARS <- expand.grid(degree = 1:2, nprune = 2:25)
set.seed(100)
MARSTune <- train(trainingAbsorp, trainingFat,
                  method = "earth",
                  tuneGrid = tuneGridMARS,
                  trControl = ctrl)


# Neural Networks
# will be very sensitive to the correlated predictors in the data set
# use PCA to preprocess
# use averaging algorithm to try and find globally minimum solution
# first make a grid of tuning params- weights and hidden units.
tuneGridNNet <- expand.grid(decay = c(0, 0.01, 0.1),
                            size = c(1:10),
                            bag = FALSE)

nnetTune <- train(x = trainingAbsorp,
                  y = trainingFat,
                  method = "avNNet",
                  preProc = c("center", "scale", "pca"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainingAbsorp) + 1) + 10 + 1,
                  tuneGrid = tuneGridNNet,
                  trControl = ctrl,
                  maxit = 500)

nnetPred <- predict(nnetTune, newdata = testingAbsorp)
postResample(pred = nnetPred, obs = testingFat)


# build a data frame of predictions, observed, then can lapply() postResample
# e.g (although exampe below is for matrix not df)
#predicted <-  matrix(rnorm(50), ncol = 5)
#observed <- rnorm(10)
#apply(predicted, 2, postResample, obs = observed)