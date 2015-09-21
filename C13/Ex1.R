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
library("reshape2")
library("earth")
library("kernlab")
library("klaR")
library("MASS")
library("mda")
library("nnet")
library("rrcov")

library("doMC")

registerDoMC(4)

data(hepatic)

# NOTES
# in future, make different reduced sets
# trees dont care about nzv, everyone else does
# guys who can do feautue selection/ dimension reduction shouldnt 
# be too offput by correlation in regression settings- but plsda performs
# better with reduced sets
# no one likes linear dependencies


# DIDNT CONSIDER ANY OTHER PREPROCESSING!! eg boxcox, spatial signs
# wonder if these could benefit??


# bio and chem contain the biological assay and chemical fingerprints
# for 281 compounds, while vecort injury contains the liver damage classification

# we have three classes for outcome

levels(injury)
table(injury)
# CHECK FOR MISSING VALUES!!!
bio[is.na(bio), ]
chem[is.na(chem), ]
sapply(bio, function(x) sum(is.na(x)))


# do some exploratory plots (perhaps consider multiplots, pca plot)
# eg
ggplot(bio, aes(x = bio[, 123])) + geom_histogram(aes(color = injury)) + geom_rug() + theme_bw()
ggplot(chem, aes(x = chem[, 123])) + geom_histogram(aes(color = injury)) + geom_rug() + theme_bw()

# we see massive class imbalance- only 30 severe cases
# use carets createDataPartition() for stratified training/ testing

# As there is exterme class imbalance, maximise Kappa rather than ROC
# also basic ROC is not defined for more than 2 classes (i am unsure
# if there are extensions that allow it in multinomial problems?)


# data splitting: caret splitting creates stratified splits
# HOW DO WE DEAL WITH CLASS IMBALANCE?
# Make a test set with only None and Mild
set.seed(123)
inTraining <- createDataPartition(injury, p = 0.8, list = FALSE)

# find the position of class "Severe" in outcome
pos <- NULL
for(i in 1:length(injury)){
  if(injury[i] == "Severe") pos <- c(pos, i)
} 

# if a severe is not in Training, add it
for(i in 1:length(pos)){
  if(!pos[i] %in% inTraining) inTraining <- c(inTraining, pos[i])
}

# rescramble order
inTraining <- sample(inTraining, length(inTraining))

injuryTrain <- injury[inTraining]
injuryTest <- injury[-inTraining]

bioTrain <- bio[inTraining, ]
bioTest <- bio[-inTraining, ]

chemTrain <- chem[inTraining, ]
chemTest <- chem[-inTraining, ]

# in terms of preprocessing, create full and reduced set
# eg for both, remove zero variance
# for reduced, remove near zero variance
# also, check in reduced that no issues related to multicolinearity
# for reduced remove high correlations (eg greater than 0.9)
# full only remove full correclation (eg greater than 0.99)
# create vectors for subsetting full set and reduced set from test and train

# check for near zero variance predictors
isNZVbio <- nearZeroVar(bioTrain, saveMetrics = TRUE, freqCut = floor(nrow(bioTrain)/5))
isNZVchem <- nearZeroVar(chemTrain, saveMetrics = TRUE, freqCut = floor(nrow(chemTrain)/5))

# full set- just remove zero var
fullSetBio <- names(bioTrain[, !isNZVbio$zeroVar])
fullSetChem <- names(chemTrain[, !isNZVchem$zeroVar])
# reduced set- remove near zero var
reducedSetBio <- names(bioTrain[, !isNZVbio$nzv])
reducedSetChem <- names(chemTrain[, !isNZVchem$nzv])

# look for high corr and virtually full corr
bioCorr <- cor(bioTrain[, fullSetBio])
highCorrBio <- findCorrelation(bioCorr, cutoff = 0.9)
fullCorrBio <- findCorrelation(bioCorr, cutoff = 0.99)
highCorrBioNames <- names(bioTrain)[highCorrBio]
fullCorrBioNames <- names(bioTrain)[fullCorrBio]

fullSetBio <- fullSetBio[!fullSetBio %in% fullCorrBioNames]
reducedSetBio <- reducedSetBio[!reducedSetBio %in% highCorrBioNames]
# have vectors for full and reduced set bio
ncol(bioTrain[, fullSetBio])
ncol(bioTrain[, reducedSetBio])

# test for collinearity (for reduced)
reducedCovMat <- cov(bioTrain[, reducedSetBio])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded

chemCorr <- cor(chemTrain[, fullSetChem])
highCorrChem <- findCorrelation(chemCorr, cutoff = 0.9)
fullCorrChem <- findCorrelation(chemCorr, cutoff = 0.99)
highCorrChemNames <- names(chemTrain)[highCorrChem]
fullCorrChemNames <- names(chemTrain)[fullCorrChem]

fullSetChem <- fullSetChem[!fullSetChem %in% fullCorrChemNames]
reducedSetChem <- reducedSetChem[!reducedSetChem %in% highCorrChemNames]
# have vectors for full and reduced set chem
ncol(chemTrain[, fullSetChem])
ncol(chemTrain[, reducedSetChem])

# test for collinearity (for reduced)
reducedCovMat <- cov(chemTrain[, reducedSetChem])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded
# see still some collinearity issues in reducedSetChem
# so fix
reducedSetChem <- reducedSetChem[!reducedSetChem %in% trimmingResults$names.discarded]
ncol(chemTrain[, reducedSetChem])

# lets have a look at pca plots for training set
chemPCA <- prcomp(chemTrain[, reducedSetChem], scale=TRUE)
type <- injuryTrain
reducedChem_pca <- ggbiplot(chemPCA, obs.scale = 1, 
                            var.scale = 1,
                            groups = type,
                            ellipse = FALSE,
                            circle = FALSE,
                            var.axes = FALSE,
                            varname.size = 3)
reducedChem_pca

bioPCA <- prcomp(bioTrain[, reducedSetBio], scale=TRUE)
type <- injuryTrain
reducedBio_pca <- ggbiplot(bioPCA, obs.scale = 1, 
                           var.scale = 1,
                           groups = type,
                           ellipse = FALSE,
                           circle = FALSE,
                           var.axes = FALSE,
                           varname.size = 3)
reducedBio_pca


# investigate some preprocessing...
# do this better with panel density plots
preProcValues <- preProcess(bioTrain[, reducedSetBio], method = c("center", "scale", "BoxCox"))

trainTransformed <- predict(preProcValues, bioTrain[, reducedSetBio])
ggplot(trainTransformed, aes(x = trainTransformed[, 33])) +
  geom_density(aes(color = injuryTrain)) + geom_rug() + theme_bw()


bioPCA <- prcomp(trainTransformed, scale=TRUE)
type <- injuryTrain
reducedBio_pca <- ggbiplot(bioPCA, obs.scale = 1, 
                           var.scale = 1,
                           groups = type,
                           ellipse = FALSE,
                           circle = FALSE,
                           var.axes = FALSE,
                           varname.size = 3)

reducedBio_pca


# I think this is going to be very challenging!
# little distinction between groups

# set up train control
# Accuracy is default for classifiers.
# 
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE)

# Now build some models for BIO predictors

# cannot do GLM for more than two outcomes
# (what about family = multinomial?)

# LDA on reduced set
# note metric is accuracy and kappa by default
# use method = "lda2" for more than two classes

# do not expect to perform well as small sample size
set.seed(476)
ldaTune <- train(x = bioTrain[, reducedSetBio],
                 y = injuryTrain,
                 method = "lda2",
                 preProc = c("center", "scale"),
                 metric = "Kappa",
                 trControl = ctrl)

confusionMatrix(ldaTune)

# testing confusion matrix
confusionMatrix(data = predict(ldaTune, newdata = bioTest[, reducedSetBio]),
                reference = injuryTest)


# PLSDA on reduced set
set.seed(476)
plsdaTune <- train(x = bioTrain[, reducedSetBio],
                   y = injuryTrain,
                   method = "pls",
                   tuneGrid = expand.grid(ncomp = 1:10),
                   preProc = c("center", "scale"),
                   probMethod = "Bayes",
                   metric = "Kappa",
                   trControl = ctrl)

confusionMatrix(plsdaTune)


# testing confusion matrix
confusionMatrix(data = predict(plsdaTune, newdata = bioTest[, reducedSetBio]),
                reference = injuryTest)

# glmnet on full set
# wonder if there is any benefit on using for reduced set?
# use family = multinomial. can you do this for plain GLM?
set.seed(476)
glmnetGrid <- expand.grid(alpha = c(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1),
                          lambda = seq(0.0, 0.04, length = 20))
glmnetTune <- train(x = bioTrain[, fullSetBio],
                    y = injuryTrain,
                    method = "glmnet",
                    tuneGrid = glmnetGrid,
                    preProc = c("center", "scale"),
                    metric = "Kappa",
                    family = "multinomial",
                    trControl = ctrl)

confusionMatrix(glmnetTune)

# hmm not very good- effectively predicting all to be mild
confusionMatrix(data = predict(glmnetTune, newdata = bioTest[, fullSetBio]),
                reference = injuryTest)

# Penalized LDA
# not sure what reasonable tuning params would be
# very slow to tune!
confusionMatrix(data = predict(penalisedldaTune, newdata = bioTest[, fullSetBio]),
                reference = injuryTest)
set.seed(476)
penldaGrid <- expand.grid(NumVars = seq(1, 141, 10),
                          lambda = 0.1)
penalisedldaTune <- train(x = bioTrain[, reducedSetBio],
                          y = injuryTrain,
                          method = "sparseLDA",
                          tuneGrid = penldaGrid,
                          preProc = c("center", "scale"),
                          metric = "Kappa",
                          trControl = ctrl)



# nearest shrunken centroids on full set
nscGrid <- data.frame(threshold = 0:20)
set.seed(476)
nscTuned <- train(x = bioTrain[, reducedSetBio],
                  y = injuryTrain,
                  method = "pam",
                  tuneGrid = nscGrid,
                  preProcess = c("center", "scale"),
                  metric = "Kappa",
                  trControl = ctrl)

# fails entirely - predicts all to be mild
# perhaps try for iris set to make sure not my error?
# works fine for iris, so maybe just data related
#nscTuned <- train(x = iris[, 1:4],
#                  y = iris$Species,
#                  method = "pam",
#                  tuneGrid = nscGrid,
#                  preProcess = c("center", "scale"),
#                  metric = "Kappa",
#                  trControl = ctrl)


confusionMatrix(data = predict(nscTuned, newdata = bioTest[, fullSetBio]),
                reference = injuryTest)




## NONLINEAR MODELS

## QDA

## NNet
# mysterious warning message?

# What the hell is going on here!

nnetGrid <- expand.grid(size = 1:7, decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$size)
set.seed(476)
nnetTune <- train(x = bioTrain[, reducedSetBio],
                  y = injuryTrain,
                  method = "nnet",
                  metric = "Kappa",
                  preProc = c("center", "scale"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 1000,
                  MaxNWts = 1*(maxSize * (length(reducedSetBio) + 1) + maxSize + 1),
                  trControl = ctrl)
plot(nnetTune)

confusionMatrix(data = predict(nnetTune, newdata = bioTest[, reducedSetBio]),
                reference = injuryTest)

## Flexible DA
# just fails?
set.seed(476)
fdaGrid <- expand.grid(nprune = 2:25,
                       degree = 1)
fdaTune <- train(x = bioTrain[, reducedSetBio],
                 y = injuryTrain,
                 method = "fda",
                 metric = "Kappa",
                 tuneGrid = fdaGrid,
                 trace = FALSE,
                 trControl = ctrl)

plot(fdaTune)

## SVM

## KNN


## Naive Bayes?


# Now build some models for CHEM predictors

set.seed(476)
ldaTuneChem <- train(x = chemTrain[, reducedSetChem],
                     y = injuryTrain,
                     method = "lda2",
                     preProc = c("center", "scale"),
                     metric = "Kappa",
                     trControl = ctrl)

# predict on test set
confusionMatrix(data = predict(ldaTuneChem, newdata = chemTest[, reducedSetChem]),
                reference = injuryTest)

# predict on training set??? probably a bad idea as overfitted to this data
ldaChemTrainProbs <- predict(ldaTuneChem, newdata = chemTrain[, reducedSetChem], type = "prob")
confusionMatrix(data = predict(ldaTuneChem, newdata = chemTrain[, reducedSetChem]),
                reference = injuryTrain)

ldaChemTrainProbs$sample <- paste(row.names(ldaChemTrainProbs), injuryTest, sep = " ")

ldaProbsMelt <- melt(ldaChemTrainProbs, id.vars = "sample")
ldaProbsMelt <- ldaProbsMelt[order(as.numeric(ldaProbsMelt$sample), decreasing = TRUE), ]

# heat plots are handy
# note that some samples are in the testing set  so missing
ggplot(ldaProbsMelt[1:90, ], aes(variable, sample)) +
  geom_tile(aes(fill = value),
            colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_bw()

# PLSDA on reduced set
set.seed(476)
plsdaTuneChem <- train(x = chemTrain[, reducedSetChem],
                       y = injuryTrain,
                       method = "pls",
                       tuneGrid = expand.grid(ncomp = 1:10),
                       preProc = c("center", "scale"),
                       probMethod = "Bayes",
                       metric = "Kappa",
                       trControl = ctrl)

confusionMatrix(plsdaTuneChem)

confusionMatrix(data = predict(plsdaTuneChem, newdata = chemTest[, reducedSetChem]),
                reference = injuryTest)

# glmnet on full set
set.seed(476)
glmnetGrid <- expand.grid(alpha = c(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1),
                          lambda = seq(0.0, 0.04, length = 20))
glmnetTuneChem <- train(x = chemTrain[, fullSetChem],
                        y = injuryTrain,
                        method = "glmnet",
                        tuneGrid = glmnetGrid,
                        preProc = c("center", "scale"),
                        metric = "Kappa",
                        family = "multinomial",
                        trControl = ctrl)

confusionMatrix(data = predict(glmnetTuneChem, newdata = chemTest[, fullSetChem]),
                reference = injuryTest)

# nearest shrunken centroids on full set
nscGrid <- data.frame(threshold = 0:20)
set.seed(476)
nscTunedChem <- train(x = chemTrain[, fullSetChem],
                      y = injuryTrain,
                      method = "pam",
                      tuneGrid = nscGrid,
                      preProcess = c("center", "scale"),
                      metric = "Kappa",
                      trControl = ctrl)

# very poor once more - predicts virtually all to be mild
confusionMatrix(data = predict(nscTunedChem, newdata = chemTest[, fullSetChem]),
                reference = injuryTest)


# Now combine BIO and CHEM and retrain same models

# probably should make a full set with zero variance predictors
# removed (for glmnet)

combinedTrain <- cbind(bioTrain, chemTrain)
combinedTest <- cbind(bioTest, chemTest)

#reducedSetCombine <- c(reducedSetBio, reducedSetChem)
#fullSetCombine <- c(fullSetBio, fullSetChem)

# check for near zero variance predictors
isNZVcomb <- nearZeroVar(combinedTrain, saveMetrics = TRUE, freqCut = floor(nrow(bioTrain)/5))

# reduced set- remove near zero var
reducedSetCombine <- names(combinedTrain[, !isNZVcomb$nzv])

# look for high corr and virtually full corr - just in reduced set
combineCorr <- cor(combinedTrain[, reducedSetCombine])
highCorrCombined <- findCorrelation(combineCorr, cutoff = 0.9)
highCorrCombinedNames <- names(combinedTrain)[highCorrCombined]

reducedSetCombine <- reducedSetCombine[!reducedSetCombine %in% highCorrCombinedNames]

# test for collinearity (for reduced)
reducedCovMat <- cov(combinedTrain[, reducedSetCombine])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded
# see still some collinearity issues in reducedSetCombine
# so fix
reducedSetCombine <- reducedSetCombine[!reducedSetCombine %in% trimmingResults$names.discarded]

combinePCA <- prcomp(combinedTrain[, reducedSetCombine], scale=TRUE)
type <- injuryTrain
reducedCombine_pca <- ggbiplot(combinePCA, obs.scale = 1, 
                               var.scale = 1,
                               groups = type,
                               ellipse = FALSE,
                               circle = FALSE,
                               var.axes = FALSE,
                               varname.size = 3)
reducedCombine_pca

# lda reduced combined set
# still complaining about colinearity??
# also very concerned that so many samples compared to predictors
# lda may be a very poor choice of model
set.seed(476)
ldaTuneCombine <- train(x = combinedTrain[, reducedSetCombine],
                        y = injuryTrain,
                        method = "lda2",
                        preProc = c("center", "scale"),
                        metric = "Kappa",
                        trControl = ctrl)


# PLSDA on reduced set
set.seed(476)
plsdaTuneCombine <- train(x = combinedTrain[, reducedSetCombine],
                          y = injuryTrain,
                          method = "pls",
                          tuneGrid = expand.grid(ncomp = 1:10),
                          preProc = c("center", "scale"),
                          probMethod = "Bayes",
                          metric = "Kappa",
                          trControl = ctrl)

confusionMatrix(data = predict(plsdaTuneCombine, newdata = combinedTest[, reducedSetCombine]),
                reference = injuryTest)

predictionProb <- predict(plsdaTuneCombine, newdata = combinedTest[, reducedSetCombine], type = "prob")
predictionClass <- predict(plsdaTuneCombine, newdata = combinedTest[, reducedSetCombine], type = "raw")

results <- cbind(predictionProb, predictionClass)
results <- cbind(results, injuryTest)



# overall, this is a tough set! 

# some comments on varImp - seems with more than 2 sets it gives varimp for each class
# I would recommend none of the models, as none can beat the no infomation rate on the test set
# I think this set requires a non linear model. for example with dimension reduction we
# do not see a clear clustering of classes (As one does, eg, for the iris set)

# so some calibration plots and lift plots (just do for combined set for plsda, as example)
# can you calibrate probabilities for more than two classes??

