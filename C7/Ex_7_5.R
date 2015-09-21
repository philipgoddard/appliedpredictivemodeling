# fit few linear from before
# fit svm (radial and poly), MARS and try nnet
# compare with postresample (do preoperly combining in data frame)
# plot comparisons- in sample and out of sample RMSE for each model
# on a needle plot?

library(reshape2)
library(pls)
library(corrplot)
library(e1071)
library(caret)
library(AppliedPredictiveModeling)
library(doMC)
registerDoMC(cores = 4)


# this would be a cool one to ocme back to once learned how to optimise
# cannot change the biological stuff, but can optimise the manufacturing process??


# If were to redo, different strategy:
# initial pre process -> only do imputation and transform outcome if needed
# (of course, still explore predictors and play with trasnforms to see what may be needed)
# then preprocess when tuning models on an as needed basis.
# can fit models (without using train) once used train to get tuning params- this may make
# them a bit easier to pull apart to get params out

data(ChemicalManufacturingProcess)

# note these are data frames, will have to convert to matrix for 
# penalty methods if use
processPredictors <- ChemicalManufacturingProcess[, -1]
yield <- ChemicalManufacturingProcess[, 1]

# quick look at outcome.
summary(yield)
# no missing

# not partic. skewed, so no need to pre process outcome
hist(yield, xlab = 'Yield')
skewness(yield)

# now look at predictors
summary(processPredictors)

# missing columns
missingCol <- sapply(processPredictors, function(x){sum(is.na(x))})
missingCol[order(missingCol, decreasing = T)]

# missing rows
missingRow <- apply(processPredictors, 1, function(x){sum(is.na(x))})
missingRow[order(missingRow,decreasing = T)]

# why does row 1 have 16 missing? and all manufacturing process have at least 1 missing
# will impute

# train test split
set.seed(517)
inTrain <- createDataPartition(yield,
                               p = 0.7,
                               list = F)

predictTrain <- processPredictors[inTrain, ]
predictTest <- processPredictors[-inTrain, ]
yieldTrain <- yield[inTrain]
yieldTest <- yield[-inTrain]

# now lets do some plotting
meltedPredict <- melt(predictTrain)

densityplot(~value | variable,
            data= meltedPredict,
            # adjust axis so measure scale different for each panel
            scales = list(x = list(relation = 'free'),
                          y = list(relation = 'free')),
            # adjust smooths
            adjust = 1.25,
            # symbol on the rug for each data point
            pch = '|',
            xlab = 'Predictor')

# impute first
trans <- preProcess(predictTrain,
                    method = c("knnImpute"))
predictTrainImp <- predict(trans, predictTrain)
predictTestImp <- predict(trans, predictTest)

# do NZV and corr after preprocess now that imputation complete
# i think it makes most sense to look at thise before start preprocessing

# near zero variance removal
ncol(predictTrainImp)
NZV <- nearZeroVar(predictTrainImp, saveMetric = T)
predictTrainImp <- predictTrainImp[, -NZV$nzv]
ncol(predictTrainImp)
predictTestImp <- predictTestImp[, -NZV$nzv]

# high correlation removal 
predCor <- cor(predictTrainImp)
highCor <- findCorrelation(predCor, cutoff = 0.9)
predictTrainImp <- predictTrainImp[, -highCor]
ncol(predictTrainImp)
predictTestImp <- predictTestImp[, -highCor]

# even after removing very high correlations, still high correlation.
# suggest that dimension reduction useful
# use PLS as I prefer to PCR!
corrplot(cor(predictTrainImp), order = 'hclust')

# preprocess -  resolve skewness, center, scale
# Surely should use yeojohnson as negative and zero. Wonder why solutions use BoxCox?

trans <- preProcess(predictTrainImp,
                    method = c("YeoJohnson", "center", "scale"))
predictTrainTrans <- predict(trans, predictTrainImp)
predictTestTrans <- predict(trans, predictTestImp)

# look again. we see that still some degenerate distributions- 
# a non-linear model may be more suitable
meltedPredictTrans <- melt(predictTrainTrans)

densityplot(~value | variable,
            data= meltedPredictTrans,
            # adjust axis so measure scale different for each panel
            scales = list(x = list(relation = 'free'),
                          y = list(relation = 'free')),
            # adjust smooths
            adjust = 1.25,
            # symbol on the rug for each data point
            pch = '|',
            xlab = 'Predictor')

# see some outliers that i do not like. Wonder if spatial sign would work?
splom(predictTrainTrans[, 1:10])


# now we have transformed data set for linear models (fit a PLS and elastic net)
# use non pre processed (and process where appropriate eg center scale for SVM) for non linear

ctrl = trainControl('boot632', repeats = 25)

# PLS
#######

set.seed(123)
plsTune <- train(x = predictTrainTrans,
                 y = yieldTrain,
                 method = 'pls',
                 trControl = ctrl,
                 tuneLength = 15)
plsTune

xyplot(RMSE ~ ncomp,
       data = plsTune$results,
       type = c("o", "g"))

plsPred <- predict(plsTune, newdata = predictTestTrans)
postResample(pred = plsPred, obs = yieldTest)

# elastic net

enetGrid <- expand.grid(lambda = c(0, 0.05, 0.1),
                        fraction = seq(0.05, 1, length = 20))

set.seed(123)
enetTune <- train(x = predictTrainTrans,
                 y = yieldTrain,
                 method = "enet",
                 tuneGrid = enetGrid,
                 trControl = ctrl)

plot(enetTune)
plot.enet(x = enetTune$finalModel)
enetFit

xyplot(RMSE ~ fraction | as.factor(lambda),
       data = enetTune$results,
       type = c("o", "g"))

enetPred <- predict(enetTune, newdata = predictTestTrans)
postResample(pred = enetPred, obs = yieldTest)

# could get final parameters by 

#### now some non linear
# use imputed data and preprocess where appropriate

# mars
# no preprocess needed
# probably would even be ok without filtered
# but... get better result with preprocess
tuneGridMARS <- expand.grid(degree = 1:2, nprune = 2:30)
set.seed(100)
MARSTuneNPP <- train(x = predictTrainImp,
                  y = yieldTrain,
                  method = "earth",
                  tuneGrid = tuneGridMARS,
                  trControl = ctrl)

MARSTuneNPP
xyplot(RMSE ~ nprune,
       data = MARSTuneNPP$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

marsnppPred <- predict(MARSTuneNPP, newdata = predictTestImp)
postResample(pred = marsnppPred, obs = yieldTest)

set.seed(100)
MARSTunePP <- train(x = predictTrainTrans,
                     y = yieldTrain,
                     method = "earth",
                     tuneGrid = tuneGridMARS,
                     trControl = ctrl)

MARSTunePP
xyplot(RMSE ~ nprune,
       data = MARSTunePP$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

marspp <- MARSTunePP$results
marspp$pp <- 'yes'
marsnpp <- MARSTuneNPP$results
marsnpp$pp <- 'no'

marsModels <- rbind(marspp, marsnpp)

xyplot(RMSE ~ nprune | pp,
       data = marsModels,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

marsppPred <- predict(MARSTuneNPP, newdata = predictTestTrans)
postResample(pred = marsppPred, obs = yieldTest)

# very odd- trasformed predictors get better in sample rmse, but out of sample 
# no pre process better RMSE and Rsquared. perhaps a cautionary tale...

# svm radial
# once again look at imputed and filtered + center scale
# versus imputed, filtered, center, scale, yeojohnson

# NPP corresponds to no yeo johnson
# PP correspondst to yeo johnson

set.seed(100)
SVMTunePP <- train(x = predictTrainTrans,
                 y = yieldTrain,
                 method = "svmRadial",
                 tuneLength = 14,
                 trControl = ctrl)

SVMTunePP

svmradialppPred <- predict(SVMTunePP, newdata = predictTestTrans)
postResample(pred = svmradialppPred, obs = yieldTest)

set.seed(100)
SVMTuneNPP <- train(x = predictTrainImp,
                   y = yieldTrain,
                   method = "svmRadial",
                   tuneLength = 14,
                   trControl = ctrl)

SVMTuneNPP

svmradialnppPred <- predict(SVMTuneNPP, newdata = predictTestImp)
postResample(pred = svmradialnppPred, obs = yieldTest)

# here is seems in test set, svm radial with yeojohnson is better than not

# svm poly
svmGrid <- expand.grid(degree = 1:4, 
                       scale = c(0.01, 0.005, 0.001), 
                       C = 2^(-2:7))

svmpolyTunePP <- train(x = predictTrainTrans,
                        y = yieldTrain,
                        method = "svmPoly",
                        tuneGrid = svmGrid,
                        trControl = ctrl)

svmpolyTunePP

xyplot(RMSE~C | as.factor(scale),
       data = svmpolyTunePP$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

svmpolyPredpp <- predict(svmpolyTunePP, newdata = predictTestTrans)
postResample(pred = svmpolyPredpp, obs = yieldTest)

# no yeo johnson
svmpolyTuneNPP <- train(x = predictTrainImp,
                       y = yieldTrain,
                       method = "svmPoly",
                       tuneGrid = svmGrid,
                       trControl = ctrl)

svmpolyTuneNPP

xyplot(RMSE~C | as.factor(scale),
       data = svmpolyTuneNPP$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

svmpolyPrednpp <- predict(svmpolyTuneNPP, newdata = predictTestImp)
postResample(pred = svmpolyPrednpp, obs = yieldTest)


svmpolyTunePP <- train(x = predictTrainTrans,
                        y = yieldTrain,
                        method = "svmPoly",
                        tuneGrid = svmGrid,
                        trControl = ctrl)

svmpolyTunePP

xyplot(RMSE~C | as.factor(scale),
       data = svmpolyTunePP$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

svmpolyPredpp <- predict(svmpolyTunePP, newdata = predictTestTrans)
postResample(pred = svmpolyPredpp, obs = yieldTest)


# results - for simplicity just look at where
# yeojohnson was applied

results <- data.frame(pls = plsPred,
                      enet = enetPred,
                      mars = marsppPred,
                      svmR = svmradialppPred,
                      svmP = svmpolyPredpp)

testTab <- vapply(results, postResample, obs = yieldTest, numeric(2))

fatResamples <- resamples(list(PLS = plsTune,
                               enet = enetTune,
                               MARS = MARSTunePP,
                               SVMr = SVMTunePP,
                               SVMp = svmpolyTunePP))

bwplot(fatResamples, metric = "RMSE")

# svm radial seems to be clear winner for non linear
# enet is far simpler than pls, so chose that for linear

plot(varImp(enetTune))
plot(varImp(SVMTunePP))

