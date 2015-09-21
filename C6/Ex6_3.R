library(reshape2)
library(pls)
library(corrplot)
library(e1071)
library(caret)
library(AppliedPredictiveModeling)

# this would be a cool one to ocme back to once learned how to optimise
# cannot change the biological stuff, but can optimise the manufacturing process??

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

# tune a model. I chose PLS
# as only 124 samples, LGOCV or bootstrap appropriate. (if many samples use 10-fold cv)
set.seed(517)
plsTune <- train(x = predictTrainTrans, y = yieldTrain,
                 method = 'pls',
                 trControl = trainControl('boot', number = 25),
                 tuneLength = 15)
plsTune

# solutions on github give some nice plots for this
# but essentially want to pull the PLS scores vs the yield for the 3 kept PLS components
plsDat <- data.frame(unclass(scores(plsTune$finalModel)))
plsDat$yield <- yieldTrain

xyplot(plsTune$results$RMSE ~ plsTune$results$ncomp,
       xlab = "# Components",
       ylab = "RMSE (Cross-Validation)",
       type = c("o", "g"))

plsPredict <- predict(plsTune, newdata = predictTestTrans)

plsResults <- data.frame(yieldActual = yieldTest, yeildPredict = plsPredict,
                         resid =  plsPredict - yieldTest)

# plots - IN SAMPLE
xyplot(yieldTrain~ predict(plsTune),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(resid(plsTune) ~ predict(plsTune),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# plots - OUT OF SAMPLE
xyplot(plsResults$yieldActual ~ plsResults$yeildPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(plsResults$resid ~ plsResults$yeildPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# out of sample RMSE and R2
# this must be luck or an error - i do not beleive out of sample RMSE should be
# better than in sample
RMSE(yieldTest, predict(plsTune, newdata = predictTestTrans))
R2(yieldTest, predict(plsTune, newdata = predictTestTrans))

# what are most important variables??
plot(varImp(plsTune, scale=F), top = 15)
topVIP <- rownames(varImp(plsTune))

# comment - small coef and low importance can be considered as candidates for removal
# look at coef(plsTune$finalModel)
# perhaps refit a model - look at lowest importance predictor with
# small coefficients (think of a clever way to do this), remove the offending ones

# explore rship between top predictors and response
plsImp <- varImp(plsTune, scale = F)
vipOrder <- order(abs(plsImp$importance), decreasing = TRUE)
topVIP <- rownames(plsImp$importance)[vipOrder[c(1:3)]]

featurePlot(predictTrainTrans[, topVIP],
            yieldTrain,
            plot = 'scatter',
            type = c('g', 'p', 'smooth'))
