library("caret")
library("AppliedPredictiveModeling")
library("nnet")
library("kernlab")
library("MASS")
library("mda")
library("earth")
library("klaR")
library("rrcov")

library("pROC")
library("rms")
library("sparseLDA")
library("subselect")
library("RColorBrewer")
library("devtools")
library("ggbiplot")
library("reshape2")
library("doMC")
library("C50")
library("dplyr")

registerDoMC(4)
data(churn)

#View(churnTrain)
summary(churnTrain)
# Maximise ROC, or kappa if poor (as strong class imbalance)

trainOutcome <- churnTrain$churn
testOutcome <- churnTest$churn
trainPred <- churnTrain[!names(churnTrain) == "churn"]
testPred <- churnTest[!names(churnTest) == "churn"]

# what is the no information rate in the test set?


##################################################
# step 1: some visualistaions

# seperate out factor, double (continuous) and integer (count) predictors
vapply(churnTrain, class, character(1))

facCols <- trainPred[, vapply(trainPred, is.factor, logical(1))]
numCols <- trainPred[, vapply(trainPred, is.double, logical(1))]
intCols <- trainPred[, vapply(trainPred, is.integer, logical(1))]

# test 
facColsT <- testPred[, vapply(testPred, is.factor, logical(1))]
numColsT <- testPred[, vapply(testPred, is.double, logical(1))]
intColsT <- testPred[, vapply(testPred, is.integer, logical(1))]


# density plots for double
colNames <- names(numCols)
j <- 1
plotListDoub <- list()
for(i in colNames){
  plt <- ggplot(numCols, aes_string(x=i)) +
    geom_density(aes(color = trainOutcome)) +
    geom_rug() +
    scale_colour_brewer(palette = "Set1") +
    theme_bw()
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListDoub[[i]] <- plt
}

multiplotList(plotListDoub, cols = 3)

# histograms for count
colNames <- names(intCols)
j <- 1
plotListInt <- list()
for(i in colNames){
  plt <- ggplot(intCols, aes_string(x = i)) +
    scale_colour_brewer(palette = "Set1") +
    geom_histogram(aes(color = trainOutcome), alpha = 0.4) +
    theme_bw()
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListInt[[i]] <- plt
}

multiplotList(plotListInt, cols = 3)

# for factors, do histograms
colNames <- names(facCols)
j <- 1
plotListFac <- list()
for(i in colNames){
  plt <- ggplot(facCols, aes_string(x = i)) +
    scale_colour_brewer(palette = "Set1") +
    geom_histogram(aes(color = trainOutcome), alpha = 0.4) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))    
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListFac[[i]] <- plt
}

multiplotList(plotListFac, cols = 2)


# We see correlations, eg between day charge and day minutes,
# which may upset models such as svm and nnet which are 
# better off with a filtered set of predictors

# do not need interaction terms for nonlinear models

##################################################
# step 3: categorical predictors to dummy
catDummies <- dummyVars(~. ,
                        data = facCols)

facTrans <- data.frame(predict(catDummies, facCols))
facTransT <- data.frame(predict(catDummies, facColsT))

# remember that there will be many pairwise collinear factors here
# as the binary yes/no will add to 1

# step 3a: remove any ZV predictors (shouldnt be any)
nzvFac <- nearZeroVar(facTrans, saveMetric = TRUE)
length(nzvFac[, nzvFac$zeroVar == TRUE])
# careful- the states have very high freq ratios... (all near zero var...)
# consider removing for reduced set

# step 3b: remove collinear (and rename)
# NOTE do we really want to remove these for all nonlinear models?
# definitely not trees! I have a feeling that fda with MARS may
# lose information as well if only have two of the three (eg can
# it split on the third that is implicit by the other two being zero?)
reducedCovMat <- cov(facTrans[, 52:58])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded

# remove and rename
facTrans <- facTrans %>%
  select(1:53, international_plan.no, voice_mail_plan.yes ) %>%
  rename(voice_mail_plan = voice_mail_plan.yes,
         international_plan = international_plan.no)

facTransT <- facTransT %>%
  select(1:53, international_plan.no, voice_mail_plan.yes ) %>%
  rename(voice_mail_plan = voice_mail_plan.yes,
         international_plan = international_plan.no)

##################################################
# step 4: process count (integer data)
# This differs form grant example, as the count variables are not 
# binned. Binning will remove information

# know that we need to transform some of these. 
# for number vmail, leave as is as nonlinear models should be
# able to deal with the imbalance
# Nonlinear models should be ok with skewed, so remove those transformations too
# for now


##################################################
# step 5: combine predictors
# do not do transformed at this stage
# not thet factTrans is just transformed to dummy
trainInput <- cbind(numCols, facTrans, intCols)
testInput <- cbind(numColsT, facTransT, intColsT)

# step 5b: same processing for test set trainInputT

# step 6: make full and reduced set

# full set: remove ZV
# reduced set: remove nzv
# NOTE trees will probably be immune to nzv predictors in state
# but for several nonlinear models (eg nnet and svm)
# uninformative predictors will upset
isNZV <- nearZeroVar(trainInput, saveMetrics = TRUE)
fullSet <- names(trainInput[, !isNZV$zeroVar])
reducedSet <- names(trainInput[, !isNZV$nzv])

# full set: correlation threshold = 0.99
# reduced set: correlation threshold = 0.9
# using the transformed intCols upsets this - WHY??
trainCorr <- cor(trainInput)
highCorr <- findCorrelation(trainCorr, cutoff = 0.9)
fullCorr <- findCorrelation(trainCorr, cutoff = 0.99)
highCorrNames <- names(trainInput)[highCorr]
fullCorrNames <- names(trainInput)[fullCorr]

fullSet <- fullSet[!fullSet %in% fullCorrNames]
reducedSet <- reducedSet[!reducedSet %in% highCorrNames]

# check for colinearity (shouldnt be any as already checked
# when made factors)
# test for collinearity (for reduced)
# once again some nonlinear models will be
# adversely effected
reducedCovMat <- cov(trainInput[, reducedSet])
trimmingResults <- trim.matrix(reducedCovMat)
trimmingResults$names.discarded


trainPCA <- prcomp(trainInput[, reducedSet], scale=TRUE)
type <- trainOutcome
reduced_pca <- ggbiplot(trainPCA, obs.scale = 1, 
                        var.scale = 1,
                        groups = type,
                        ellipse = TRUE,
                        circle = TRUE,
                        var.axes = FALSE,
                        varname.size = 3)
reduced_pca + theme_bw() + scale_colour_brewer(palette = "Set1") 

# now 

# step 7: fit models
# 7a: train control
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)

# regularized DA (mix of lda and qda)
# tune over lambda and gamma
set.seed(476)
qdaTune <- train(x = trainInput[, reducedSet],
                 y = trainOutcome,
                 method = "rda",
                 preProc = c("center", "scale"),
                 metric = "Sens",
                 trControl = ctrl)

#save(qdaTune, file ="qdaChurn.RData")

# with optimum gamma = 1 and lambda = 0 this corresponds to 
# LDA using (lambda) with independent covariance matrix (gamma = 0)

# Mixture discriminant analysis
# tuning over subclasses per class
mdaGrid <- expand.grid(subclasses = 1:12)
set.seed(476)
mdaTune <- train(x = trainInput[, reducedSet],
                 y = trainOutcome,
                 method = "mda",
                 preProc = c("center", "scale"),
                 metric = "Sens",
                 tuneGrid = mdaGrid,
                 trControl = ctrl)

# hmm seems optimal at 12 subclasses? plus warning messages...
# risk of overfit
#save(mdaTune, file ="mdaChurn.RData")


## nnet
nnetGrid <- expand.grid(size = 1:10,
                        decay = c(0, 0.1, 1, 2))
maxSize <- max(nnetGrid$size)
numWts <- 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1)
set.seed(476)
nnetTune <- train(x = trainInput[, reducedSet],
                  y = trainOutcome,
                  method = "nnet",
                  metric = "Sens",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 1000,
                  MaxNWts = numWts,
                  trControl = ctrl)

# may want to consider more hidden units(1:14), focus on decay = 0, 0.05, 0.1
#save(nnetTune, file ="nnetChurn.RData")

nnetPred <- predict(nnetTune, newdata = testInput[, reducedSet])
confusionMatrix(data = nnetPred, reference = testOutcome)

# fda with MARS
marsGrid <- expand.grid(degree = 1,
                        nprune = seq(2, 40, 2))
set.seed(476)
fdaTune <- train(x = trainInput[, fullSet],
                 y = trainOutcome,
                 method = "fda",
                 metric = "Sens",
                 preProc = c("center", "scale"),
                 tuneGrid = marsGrid,
                 trControl = ctrl)
#save(fdaTune, file ="fdaChurn.RData")


# best at nprune = 28
# for fit coefficients- look at summary(fdaTune$finalModel$fit)

# svm Radial basis
# fit on reduced set as uninformative 
# predictors may impair
set.seed(202)
# estimate sigma
sigmaRangeReduced <- sigest(as.matrix(trainInput[, reducedSet]))
svmRGridReduced <- expand.grid(sigma = sigmaRangeReduced[1],
                               C = 2^(seq(-4, 4)))
set.seed(476)
svmRTune <- train(x = trainInput[, reducedSet],
                  y = trainOutcome,
                  method = "svmRadial",
                  metric = "Sens",
                  preProc = c("center", "scale"),
                  tuneGrid = svmRGridReduced,
                  fit = FALSE,
                  trControl = ctrl)

#save(svmRTune, file ="svmRChurn.RData")


# something changed... not sure what :S
# now get sigma = 0.023
# very competitive- looks like nnet and svm are top two contendors here
# C = 0.25, sigma = 0.01752151, ROC = 0.8922, Sens = 0.653, spec = 0.9597

svmPred <- predict(svmRTune, newdata = testInput[, reducedSet])
confusionMatrix(data = svmPred, reference = testOutcome)

# k-nn
set.seed(476)
knnFit <- train(x = trainInput[, reducedSet],
                y = trainOutcome,
                method = "knn",
                metric = "Sens",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(k = seq(3, 50, 2)),
                trControl = ctrl)

#save(knnFit, file ="knnChurn.RData")

# For naive bayes, need data fed in as categorical, not dummy
# skip for now- unlikely to outperform svm or nnet

# load("~/R/appliedpredictivemodeling/C13/qdaChurn.RData")
# load("~/R/appliedpredictivemodeling/C13/mdaChurn.RData")
# load("~/R/appliedpredictivemodeling/C13/nnetChurn.RData")
# load("~/R/appliedpredictivemodeling/C13/fdaChurn.RData")
# load("~/R/appliedpredictivemodeling/C13/svmRChurn.RData")
# load("~/R/appliedpredictivemodeling/C13/knnChurn.RData")



models <- list(rda = qdaTune,
               mda = mdaTune,
               nnet = nnetTune,
               fda = fdaTune,
               svm = svmRTune,
               knn = knnFit)

inputSet <- list(rdaData = testInput[, reducedSet],
             mdaData = testInput[, reducedSet],
             nnetData = testInput[, reducedSet],
             fdaData = testInput[, fullSet],
             svmData = testInput[, reducedSet],
             knnData = testInput[, reducedSet])

predictions <- Map(function(x, y) {
  out <- list()
  out[[1]] <- predict(x, newdata = y)
  out[[2]] <- predict(x, newdata = y, type = "prob")$yes
  out[[3]] <- confusionMatrix(data = predict(x, newdata = y), reference = testOutcome)
  out
  },
  models,
  inputSet)

results <- do.call(cbind.data.frame, lapply(predictions, "[[", 2))
results$class <- testOutcome
                   

calCurve <- calibration(class ~  svm + nnet , data = results)
calCurve
xyplot(calCurve,
       auto.key = list(columns = 3))

# svm and nnet well calibrated
# fda, rda, mda needs calibrating
# knn worthless

liftCurve <- lift(class ~ svm + nnet, data = results)
liftCurve

xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = T,
                       points = F))

# calibrations

modelstoCal <- select(results, rda, mda, fda, class)
calibrationModels <- lapply(names(modelstoCal)[1:3],
                     function(colName){
                       model <- NaiveBayes(reformulate(termlabels = colName, response = 'class'),
                                  data = modelstoCal,
                                  usekernel = TRUE)})

names(calibrationModels)[1:3] <- c("rda", "mda", "fda" )

calibrationResults <- Map(
  function(x, y){
    out <- list()
    out[[1]] <- predict(x, newdata = as.data.frame(y))$posterior[, "yes"]
    out[[2]] <- predict(x, newdata = as.data.frame(y))$class
    out[[3]] <- confusionMatrix(data = predict(x, newdata = as.data.frame(y))$class,
                                reference = testOutcome)
    out
    },
  calibrationModels,
  modelstoCal[, 1:3]
  )

calibrated <- do.call(cbind.data.frame, lapply(calibrationResults, '[[', 1))
calibrated$class <- testOutcome

calCurve <- calibration(class ~  rda + mda + fda , data = calibrated)
calCurve
xyplot(calCurve,
       auto.key = list(columns = 3))


# fda and mda calibration seems improved. rda no good.
# so for final models to present:
# svm, nnet, mdaCal and fdaCal
# in terms of kappa we see that calibraration has negative effect
# apart from on mda


## MAKE THIS DRY BEFORE SHARE WITH THE WORLD
KappasCal <- do.call(rbind.data.frame,(lapply(lapply(calibrationResults, '[[', 3), function(x) x$overall[[2]])))
KappasCal$model <- rownames(KappasCal)
names(KappasCal)[1] <- "kappa"
KappasCal$type <- rep("cal", 3)

Kappas <- do.call(rbind.data.frame,(lapply(lapply(predictions, '[[', 3), function(x) x$overall[[2]])))
Kappas$model <- rownames(Kappas)
names(Kappas)[1] <- "kappa"
Kappas$type <- rep("noCal", nrow(Kappas))

Kappas <- rbind(Kappas, KappasCal)

plot1 <- ggplot(Kappas, aes(x = kappa,
                            y = reorder(model, kappa),
                            color = type,
                            shape = type), ) +
  geom_point(alpha = 0.6, size = 5) +
  scale_shape_manual(values=c(16, 17)) +
  scale_colour_brewer(palette="Set1") +
  scale_x_continuous(limits = c(0.3, 0.7)) +
  theme_bw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dashed"),
        legend.position = c("top"),
        text = element_text(size=11),
        legend.title = element_blank(),
        legend.key = element_blank() ) +
  xlab('Kappa') +
  ylab('Model') + 
  ggtitle("Kappa of selected models") 

## accuracy
## MAKE THIS DRY BEFORE SHARE WITH THE WORLD
AccCal <- do.call(rbind.data.frame,(lapply(lapply(calibrationResults, '[[', 3), function(x) x$overall[[1]])))
AccCal$model <- rownames(AccCal)
names(AccCal)[1] <- "Accuracy"
AccCal$type <- rep("cal", 3)

Acc <- do.call(rbind.data.frame,(lapply(lapply(predictions, '[[', 3), function(x) x$overall[[1]])))
Acc$model <- rownames(Acc)
names(Acc)[1] <- "Accuracy"
Acc$type <- rep("noCal", nrow(Acc))

Acc <- rbind(Acc, AccCal)

plot2 <- ggplot(Acc, aes(x = Accuracy,
                            y = reorder(model, Accuracy),
                            color = type,
                            shape = type), ) +
  geom_point(alpha = 0.6, size = 5) +
  scale_shape_manual(values=c(16, 17)) +
  scale_colour_brewer(palette="Set1") +
  scale_x_continuous(limits = c(0.8, 1)) +
  theme_bw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dashed"),
        legend.position = c("top"),
        text = element_text(size=11),
        legend.title = element_blank(),
        legend.key = element_blank() ) +
  geom_vline(aes(xintercept = 0.8656), linetype = 'dashed') +
  xlab('Accuracy') +
  ylab('Model') + 
  ggtitle("Accuracy of selected models") 



# AUC for ROC
# bare in mind that tuned models for Sens, not ROC or kappa... 
## MAKE THIS DRY BEFORE SHARE WITH THE WORLD
rocCurves <- lapply(results[,!names(results)=="class"],
                    function(x) roc(response = testOutcome,
                                    predictor = x,
                                    levels = rev(levels(results$class))))

rocCurvesCal <- lapply(calibrated[,!names(calibrated)=="class"],
                    function(x) roc(response = testOutcome,
                                    predictor = x,
                                    levels = rev(levels(calibrated$class))))

aucROC <- lapply(rocCurves, function(x) auc(x)[[1]])
calaucROC <- lapply(rocCurvesCal, function(x) auc(x)[[1]])

dfAUC <- do.call(rbind.data.frame, aucROC)
dfAUC$model <- row.names(dfAUC)
dfAUCcal <- do.call(rbind.data.frame, calaucROC)
dfAUCcal$model <- row.names(dfAUCcal)

dfAUC$type <- rep(c("noCal"), 6)
dfAUCcal$type <- rep(c("Cal"), 3)

names(dfAUC)[1] <- "ROC"
rownames(dfAUC) <- NULL
names(dfAUCcal)[1] <- "ROC"
rownames(dfAUCcal) <- NULL

dfAUC <- rbind(dfAUC, dfAUCcal)



plot3 <- ggplot(dfAUC, aes(x = ROC,
                           y = reorder(model, ROC),
                           color = type,
                           shape = type) ) +
  geom_point(alpha = 0.6, size = 5) +
  scale_shape_manual(values=c(16, 17)) +
  scale_colour_brewer(palette="Set1") +
  scale_x_continuous(limits = c(0.7, .95)) +
  theme_bw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dashed"),
        legend.position = c("top"),
        text = element_text(size=11),
        legend.title = element_blank(),
        legend.key = element_blank() ) +
  xlab('ROC') +
  ylab('Model') + 
  ggtitle("AUC of ROC curves for selected models") 



#####################################################
multiplotList <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  plots <-  c(..., plotlist)
  numPlots = length(plots)
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

########################################################
quadInteraction <- function(df){
  stopifnot(is.data.frame(df))
  numericCols <- vapply(df, is.numeric, logical(1)) 
  dfNum <- df[, numericCols]
  nCols <- ncol(df) + choose(ncol(dfNum), 2) + ncol(dfNum)
  
  out <- data.frame(matrix(NA, nrow = nrow(df), ncol = nCols))
  out[1:ncol(df)] <- df
  names(out)[1:ncol(df)] = names(df)
  
  start <- 1
  dfPosition <- ncol(df) + 1
  for( i in 1:ncol(dfNum) ) {
    for( j in start:ncol(dfNum) ) {
      out[dfPosition] <- dfNum[i] * dfNum[j]
      names(out)[dfPosition] <- paste(names(dfNum[i]),
                                      names(dfNum[j]),
                                      sep = ' * ')
      dfPosition <- dfPosition + 1
    }
    start <- start + 1 
  }
  out
}


#########################################################
replace <- function(x, curr, repl, regex = FALSE, ...) {
  stopifnot(length(curr) == length(repl))
  stopifnot(is.character(curr) && is.character(repl))
  
  for(i in 1:length(curr)){ 
    # if(regex){
    x <- gsub(curr[i], repl[i], x, ...)
    #  } else {
    #    x[x == curr[i] ] <- repl[i]
    #  }
  }
  x
}

############################################################
mround <- function(x, base){ 
  base * round(x / base) 
} 