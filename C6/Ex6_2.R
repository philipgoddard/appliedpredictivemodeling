# Q 6.2

library(elasticnet)
library(pls)
library(corrplot)
library(caret)
library(AppliedPredictiveModeling)

data(permeability)

head(fingerprints)
head(permeability)

ncol(fingerprints)
# 1107 columns

NZV <- nearZeroVar(fingerprints, saveMetric = T)
fingerprintsFilt <- fingerprints[, !NZV$nzv]

ncol(fingerprintsFilt)
# 388 columns

# practice plotting to make this nicer
# see that a transform of permeability will be needed to obviate skew
hist(permeability)
hist(log10(permeability))

set.seed(101)
inTrain <- createDataPartition(permeability, p = 0.75, list=F)
# 10 fold cv may not be as appropriate as fairly small population
# suggest that LOOGCV may be more optimal
# this differs from k-fold as samples can be represented in multiple held-out
# subsets
ctrl <- trainControl(method = 'LGOCV')

PermTrain <- permeability[inTrain, ]
PermTest <- permeability[-inTrain, ]
FingFiltTrain <- fingerprintsFilt[inTrain, ]
FingFiltTest <- fingerprintsFilt[-inTrain, ]

PLSTune <- train(x = FingFiltTrain, y = log10(PermTrain), 
                method = 'pls',
                trControl = ctrl,
                tuneGrid = expand.grid(ncomp = 1:15),
                preProcess = c('center', 'scale'))

xyplot(PLSTune$results$RMSE ~ PLSTune$results$ncomp,
       type = c("p", "g", "l"),
       xlab = "ncomp", ylab = "RMSE")
xyplot(PLSTune$results$Rsquared ~ PLSTune$results$ncomp,
       type = c("p", "g", "l"),
       xlab = "ncomp", ylab = "RMSE")

PLSTune

# 9 latent variables needed for optimum RMSE, 
# 0.49(0.09) optimum RMSE
# 0.49(0.14) optimum R2

# 

PLSPredict <- predict(PLSTune, newdata = FingFiltTest)
PLSResults <- data.frame(permActual = log10(PermTest), permPredict = PLSPredict,
                         resid =  PLSPredict - log10(PermTest))

RMSE(PLSResults$permPredict, PLSResults$permActual)
R2(PLSResults$permPredict, PLSResults$permActual, formula = "corr")

# would be of interest to plot first few components vs response to confirm linear r/ship
# (have a look at solutions for a nice way to do this- need to extract and plot)

# plot prediction vs actual, and residuals vs predictions
# out of sample
xyplot(PLSResults$permActual ~ PLSResults$permPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(PLSResults$resid ~ PLSResults$permPredict,
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

# other models - ols may struggle if correlated predictors
# pcr probaly wont be as good as pls
# so enet is a good choice to try

enetGrid <- expand.grid(lambda = c(0, 0.05, 0.1),
                        fraction = seq(0.05, 1, length = 20))

# why does lgocv fail here? continue from here
ctrl <- trainControl(method = 'LGOCV')

set.seed(614)
enetFit <- train(x = FingFiltTrain, y = log10(PermTrain),
                 method = "enet",
                 tuneGrid = enetGrid,
                 trControl = ctrl)

plot(enetFit)
plot.enet(x = enetFit$finalModel)
enetFit

# how get kept predictors out?? most odd. why does coef(enetFit$finalmodel) not work?

enetPredict <- predict(enetFit, newdata = FingFiltTest)
enetResults <- data.frame(permActual = log10(PermTest), permPredict = enetPredict,
                         resid =  enetPredict - log10(PermTest))

RMSE(enetResults$permPredict, enetResults$permActual)
R2(enetResults$permPredict, enetResults$permActual, formula = "corr")


# tune a ridge regression as well

set.seed(614)
ridgeGrid <- expand.grid(lambda = seq(0.02, 0.4, length = 9))
ridgeTune <- train(x = FingFiltTrain, y = log10(PermTrain),
                   method = "ridge",
                   tuneGrid = ridgeGrid,
                   trControl = ctrl)

ridgeTune

plot(ridgeTune$results$Rsquared ~ ridgeTune$results$lambda, type = 'l')
points(ridgeTune$results$Rsquared ~ ridgeTune$results$lambda)

ridgePredict <- predict(ridgeTune, newdata = FingFiltTest)
ridgeResults <- data.frame(permActual = log10(PermTest), permPredict = ridgePredict,
                          resid =  ridgePredict - log10(PermTest))

RMSE(ridgeResults$permPredict, ridgeResults$permActual)
R2(ridgeResults$permPredict, ridgeResults$permActual, formula = "corr")


# enet has lower rmse than ridge- probabbly the best penalty method to use
# it outperforms PLS as well

# note could not use OLS as 


# to do - discuss and compare results
# extract PLS components and plot first few
# tune elastic net properly
# plot fraction of full solution stuff to investigate what parameters removed

# Note- weird. in this case when training PLS it chose the best model (based in RMSE)
# in previous excercise it was acting odd. Perhaps 
