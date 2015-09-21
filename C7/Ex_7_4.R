library(corrplot)
library(earth)
library(caret)
library(AppliedPredictiveModeling)
library(doMC)
registerDoMC(cores = 4)

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
# suggest that LGOOCV may be more optimal
# this differs from k-fold as samples can be represented in multiple held-out
# subsets
ctrl <- trainControl(method = 'LGOCV')

PermTrain <- permeability[inTrain, ]
PermTest <- permeability[-inTrain, ]
FingFiltTrain <- fingerprintsFilt[inTrain, ]
FingFiltTest <- fingerprintsFilt[-inTrain, ]

# train a nnet, mars, svm (poly and radial), and knn
# note no need to preprocess, as binary fingerprints

# knn (note only use odd k's in classifier with
# 2 groups to avoid ties)
tuneGridKNN <- data.frame(k = 1:20)
set.seed(100)
KNNTune <- train(x = FingFiltTrain,
                 y = log10(PermTrain),
                 method = "knn",
                 #preProc = c("center", "scale"),
                 tuneGrid = tuneGridKNN,
                 trControl = ctrl)

KNNTune
xyplot(RMSE ~ k,
       data = KNNTune$results,
       type = c('o', 'g'))

knnPred <- predict(KNNTune, newdata = FingFiltTest)
postResample(pred = knnPred, obs = log10(PermTest))

# mars
set.seed(100)
# using this approach may not be best for estimate as 
# no resampling
MARSfit <- earth(FingFiltTrain, PermTrain)
summary(MARSfit)
# get an idea from this that number of terms sensible to tune over
# 2:20 seems reasonable

tuneGridMARS <- expand.grid(degree = 1:2, nprune = 2:20)
set.seed(100)
MARSTune <- train(x = FingFiltTrain, 
                  y = log10(PermTrain),
                  method = "earth",
                  tuneGrid = tuneGridMARS,
                  trControl = ctrl)

MARSTune
xyplot(RMSE ~ nprune,
       data = MARSTune$results,
       type = c('o', 'g'),
       groups = degree,
       auto.key = list(columns = 2))

marsPred <- predict(MARSTune, newdata = FingFiltTest)
postResample(pred = marsPred, obs = log10(PermTest))

# svm (radial)
# use default cost tune grid
set.seed(100)
SVMTune <- train(x = FingFiltTrain, 
                 y = log10(PermTrain),
                 method = "svmRadial",
                 #preProc = c("center", "scale"),
                 tuneLength = 14,
                 trControl = ctrl)

SVMTune

svmradialPred <- predict(SVMTune, newdata = FingFiltTest)
postResample(pred = svmradialPred, obs = log10(PermTest))

#nnet - slooow. cant be bothered to run
tuneGridNNet <- expand.grid(decay = c(0, 0.01, 0.1),
                            size = c(1:10),
                            bag = FALSE)

# nnetTune <- train(x = FingFiltTrain,
                  y = log10(PermTrain),
                  method = "avNNet",
                 # preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(FingFiltTrain) + 1) + 10 + 1,
                  tuneGrid = tuneGridNNet,
                  trControl = ctrl,
                  maxit = 500)

nnetTune
nnetPred <- predict(nnetTune, newdata = FingFiltTest)
postResample(pred = nnetPred, obs = log10(PermTest))

