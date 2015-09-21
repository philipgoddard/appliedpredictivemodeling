library("caret")
library("AppliedPredictiveModeling")
library("glmnet")
library("MASS")
library("pamr")
library("pls")
library("pROC")
library("rms")
library("sparseLDA")
library("subselect")
library("RColorBrewer")
library("devtools")
library("ggbiplot")
library("doMC")
library("kernlab")


registerDoMC(4)

data(oil)
str(oilType)
qplot(oilType)
table(oilType)

# huge class imbalance.
# make a test set containing only types A and B
set.seed(123)
inTraining <- createDataPartition(oilType, p = 0.8, list = FALSE)

exclude <- c("C", "D", "E", "F", "G")
gPos <- c("G")

# find the position of class to exclude in test set
pos <- NULL
posii <- NULL
for(i in 1:length(oilType)){
  if(oilType[i] %in% exclude) pos <- c(pos, i)
  if(oilType[i] %in% gPos) posii <- c(posii, i)
} 

# if a excluded types not in Training, add them
for(i in 1:length(pos)){
  if(!pos[i] %in% inTraining) inTraining <- c(inTraining, pos[i])
}


# rescramble order
inTraining <- sample(inTraining, length(inTraining))

oilTrain <- oilType[inTraining]
oilTest <- oilType[-inTraining]

fatTrain <- fattyAcids[inTraining, ]
fatTest <- fattyAcids[-inTraining, ]

# appropriate to use LOOCV as only 84 samples in train. Also will ensure
# that samples with only 2 in will be chosen in resampling groups

# also optimise Kappa statistic due to class imbalance

# assume (blindly) that no nZV or collinearty issues

# PCA plot
fatPCA <- prcomp(fatTrain, scale=TRUE)
type <- oilTrain
fatPCA_plot <- ggbiplot(fatPCA, obs.scale = 1, 
                        var.scale = 1,
                        groups = type,
                        ellipse = FALSE,
                        circle = FALSE,
                        var.axes = FALSE,
                        varname.size = 3)
fatPCA_plot + theme_bw()

# use LOOCV
ctrl <- trainControl(method = 'repeatedcv',
                     #method = "LOOCV",
                     classProbs = TRUE,
                     savePredictions = TRUE)

# for LOOCV no SD in kappa or accuracy because only one
# prediction for each hold out sample!

# optimise kappa statistic as 

set.seed(476)
ldaTune <- train(x = fatTrain,
                 y = oilTrain,
                 method = "lda2",
                 preProc = c("center", "scale"),
                 metric = "Kappa",
                 trControl = ctrl)


# testing confusion matrix
# Remember, only groups A and B were in the testing sample
# due to class imbalance

confusionMatrix(data = predict(ldaTune, newdata = fatTest),
                reference = oilTest)

# training: optimal tune gets kappa = 0.954
# testing: get kappa = 1, but remeber only approximate 
# measure of performance due to only two classes in testing sample

set.seed(476)
plsdaTune <- train(x = fatTrain,
                   y = oilTrain,
                   method = "pls",
                   tuneGrid = expand.grid(ncomp = 1:5),
                   preProc = c("center", "scale"),
                   probMethod = "Bayes",
                   metric = "Kappa",
                   trControl = ctrl)
# when using 10fold cv get warning that missing values in resampled performance measures

confusionMatrix(data = predict(plsdaTune, newdata = fatTest),
                reference = oilTest)

# plsda model fails - i guess due to class G having only 2 points
# get :
# task 23 failed - "need at least 2 points to select a bandwidth automatically"
# where one point is in holdout in trainin? Perhaps need to eliminate class G from 
# model

# glmnet
set.seed(476)
glmnetGrid <- expand.grid(alpha = c(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1),
                          lambda = seq(0.0, 0.1, length = 10))
glmnetTune <- train(x = fatTrain,
                    y = oilTrain,
                    method = "glmnet",
                    tuneGrid = glmnetGrid,
                    preProc = c("center", "scale"),
                    metric = "Kappa",
                    family = "multinomial",
                    trControl = ctrl)

# get same (i think) problem as plsda:
# Error in { : 
# task 177 failed - "one multinomial or binomial class has 1 or 0 observations; not allowed"
# perhaps we should remove class G?

# penalised lda
penldaGrid <- expand.grid(NumVars = 1:7,
                          lambda = seq(0.0, 0.4, 0.05))

penalisedldaTune <- train(x = fatTrain,
                          y = oilTrain,
                          method = "sparseLDA",
                          tuneGrid = penldaGrid,
                          preProc = c("center", "scale"),
                          metric = "Kappa",
                          trControl = ctrl)

# doesnt seem to hold benfit over non- penalised version

# nearest shrunken centroids
nscGrid <- data.frame(threshold = 0:20)
set.seed(476)
nscTuned <- train(x = fatTrain,
                  y = oilTrain,
                  method = "pam",
                  tuneGrid = nscGrid,
                  preProcess = c("center", "scale"),
                  metric = "Kappa",
                  trControl = ctrl)

confusionMatrix(data = predict(nscTuned, newdata = fatTest),
                reference = oilTest)


# svm radial
# fails with loocv??

# estimate sigma
sigmaRangeReduced <- sigest(as.matrix(fatTrain))
svmRGridReduced <- expand.grid(sigma = sigmaRangeReduced[1],
                               C = 2 ^ (seq(-4, 4)))
#dislikes repeated cv
set.seed(476)
svmRTune <- train(x = fatTrain,
                  y = oilTrain,
                  method = "svmRadial",
                  metric = "Kappa",
                  preProc = c("center", "scale"),
                  tuneGrid = svmRGridReduced,
                  fit = FALSE,
                  trControl = trainControl(method='LOOCV'))

# nnet

nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$size)
set.seed(476)
nnetTune <- train(x = fatTrain,
                  y = oilTrain,
                  method = "nnet",
                  metric = "Kappa",
                  preProc = c("center", "scale"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 2000,
                  MaxNWts = 1*(maxSize * (length(oilTrain) + 1) + maxSize + 1),
                  trControl = ctrl)
plot(nnetTune)

# i dont get why different models seem grumpy with different resampling choices?....
# plsda refuses for loocv, as does svm. but surely 10 fold cv
# has problems as possible that class g will not be present in some resamples?
