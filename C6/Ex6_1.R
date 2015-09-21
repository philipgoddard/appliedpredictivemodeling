# Q6.1
library(pls)
library(corrplot)
library(caret)
data(tecator)

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

pcaObject <- prcomp(absorp, center=TRUE, scale. = TRUE)
percentVariance <- pcaObject$sdev^2/sum(pcaObject$sdev^2)*100
percentVariance[1:3]

#scree plot
plot(pcaObject$sdev, ylab = 'percent variance', xlab = 'component')
lines(pcaObject$sdev)

# first few pca components
transparentTheme(pchSize = .8, trans = .3)
panelRange <- extendrange(pcaObject$x[, 1:3])
splom(as.data.frame(pcaObject$x[, 1:3]),
     # groups = segTrainClass,
      type = c("p", "g"),
      as.table = TRUE,
      auto.key = list(columns = 2),
      prepanel.limits = function(x) panelRange)

# see huge correlation, so little surpise only 1 PC is needed
absorpCor <- cor(absorp)
corrplot(absorpCor, order='hclust')

# what is effective dimension of data?
# essentially all the variance is held in the first componont

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

# set up cross validation method - use 10 fold cv as large set
ctrl <- trainControl(method = 'cv', number = 10)

# OLS first
# no preprocessing, as want a nice, parsimonious model
# which can be used for explaining rships

lmFit <- train(x = trainingAbsorp, y = trainingFat, 
               method = 'lm', trControl = ctrl)

# have a lool at lmFit$resample
# lmFit$results
# have in sample RMSE = 5.227 with SD of 0.899
# have in sample RSquared = 4.719 with sd of 0.115

# look at summary(lmFit) for final params- see that many are not
# statistically significant

# look at test set now
lmPredict <- predict(lmFit, testingAbsorp)
# make a df of tests and predictions
lmResults <- data.frame(fatActual = testingFat, fatPredict = lmPredict,
                        resid =  lmPredict - testingFat)

# plots - IN SAMPLE
xyplot(trainingFat ~ predict(lmFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(resid(lmFit) ~ predict(lmFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(lmResults$fatActual ~ lmResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(lmResults$resid ~ lmResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# out of sample RMSE and R2
RMSE(lmResults$fatPredict, lmResults$fatActual)
R2(lmResults$fatPredict, lmResults$fatActual) #huh? why started kicking error? Gets masked by pls 
# use PostResample() function here instead!!
# get RMSE = 3.286
# get R2 = 0.9424

# PCR
# first build PCA transform on training set
# then apply same transform on testing

pcaObject <- prcomp(trainingAbsorp, center=TRUE, scale. = TRUE)
trainingAbsorpTrans <- predict(pcaObject, trainingAbsorp)
testingAbsorpTrans <- predict(pcaObject, testingAbsorp)

# NOTE - seems that you dont have to put a transformed object 
# into here (could have just used trainingAbsorp)
PCRFit <- train(x = trainingAbsorpTrans, y = trainingFat, 
               method = 'pcr',
               trControl = ctrl,
               tuneGrid = expand.grid(ncomp = 1:100))

PCRFit$results
PCRFit$bestTune

# we see best tune has n=20, RMSE = 2.45(0.757) and R2 = 0.966(0.0176)

# have a look at RMSE from tuning over number of components
xyplot(PCRFit$results$RMSE ~ PCRFit$results$ncomp,
       type = c("p", "g", "l"),
       xlab = "ncomp", ylab = "RMSE")

# does this take the optimum number of components by default?
PCRPredict <- predict(PCRFit, newdata = testingAbsorpTrans)

PCRResults <- data.frame(fatActual = testingFat, fatPredict = PCRPredict,
                        resid =  PCRPredict - testingFat)

# plots - IN SAMPLE
xyplot(trainingFat ~ predict(PCRFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
#pcr very odd here - plots resids for up to 32 components?? how specify the best?
xyplot(resid(PCRFit) ~ predict(PCRFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(PCRResults$fatActual ~ PCRResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(PCRResults$resid ~ PCRResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")
# some hints of not constant variance

# out of sample RMSE and R2
RMSE(PCRResults$fatPredict, PCRResults$fatActual)
R2(PCRResults$fatPredict, PCRResults$fatActual) ## huh? why not work??
# out of sample RMSE is 2.437
# out of sample R2 is 

# PLS
PLSFit <- train(x = trainingAbsorp, y = trainingFat, 
                method = 'pls',
                trControl = ctrl,
                tuneGrid = expand.grid(ncomp = 1:100),
                preProcess = c('center', 'scale'))

PLSFit
PLSFit$results
PLSFit$bestTune

# plot PLS tuning compared to PCR
plsResamples <- PLSFit$results
plsResamples$Model <- "PLS"
pcrResamples <- PCRFit$results
pcrResamples$Model <- "PCR"
plsPlotData <- rbind(plsResamples, pcrResamples)

xyplot(RMSE ~ ncomp,
       data = plsPlotData,
       #aspect = 1,
       xlab = "# Components",
       ylab = "RMSE (Cross-Validation)",
       auto.key = list(columns = 2),
       groups = Model,
       type = c("o", "g"))

# test
PLSPredict <- predict(PLSFit, newdata = testingAbsorp)

PLSResults <- data.frame(fatActual = testingFat, fatPredict = PLSPredict,
                         resid =  PLSPredict - testingFat)

# should make plots to check nothing funny going on with residuals etc

# plots - IN SAMPLE
xyplot(trainingFat ~ predict(PLSFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")

# something odd here- is resids for fits for first 18 components
xyplot(resid(PLSFit) ~ predict(PLSFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(PLSResults$fatActual ~ PLSResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(PLSResults$resid ~ PLSResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# out of sample RMSE and R2
RMSE(PLSResults$fatPredict, PLSResults$fatActual)
R2(PLSResults$fatPredict, PLSResults$fatActual)
# out of sample RMSE is 2.44
# out of sample R2 is 0.96

# note- PLS needed only 20 components, PCR needed 20
# despite the fact that PLS is margionally better for RMSE, it is simpler model

# elastic net

# set grid
# write some notes explainig! fraction is fraction of full (ols) solution, lambda is lambda
# for the 'ridge' part
enetGrid <- expand.grid(.lambda = c(0, 0.001, 0.01, 0.1, 1),
                        .fraction = seq(0.05, 1, length = 20))

# note predictors MUST be a matrix for enet
enetFit <- train(trainingAbsorp, trainingFat,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))

plot(enetTune)
enetFit

# the optimal model has lambda1 = 0 eg is lasso

# test
enetPredict <- predict(enetFit, newdata = testingAbsorp)

enetResults <- data.frame(fatActual = testingFat, fatPredict = enetPredict,
                         resid =  enetPredict - testingFat)


# plots - IN SAMPLE
xyplot(trainingFat ~ predict(enetFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")

xyplot(resid(enetFit) ~ predict(enetFit),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(enetResults$fatActual ~ enetResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(enetResults$resid ~ enetResults$fatPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# notes - PLS suitable as correlated predictors- fares better than OLS
# PLS needs less components than PCR
# PLS would prabably be preferred to lasso as suitable for highly correlated data

# summarise and compare!
