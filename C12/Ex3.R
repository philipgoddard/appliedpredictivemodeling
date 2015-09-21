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
library("doMC")
library("C50")
library("dplyr")
library("klaR")

registerDoMC(4)
data(churn)


#View(churnTrain)
summary(churnTrain)
# Maximise ROC, or kappa if poor (as strong class imbalance)

trainOutcome <- churnTrain$churn
testOutcome <- churnTest$churn
trainPred <- churnTrain[!names(churnTrain) == "churn"]
testPred <- churnTest[!names(churnTest) == "churn"]


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
# which also display degenerate distrubutions that may upset
# linear models.
# Consider transformations to fix this?

# Also see that total intl calls is very skewed
# number_vmail_messages has very odd distribution
# number_customer_service_calls also skewed

##################################################
# step 2: seperate continuous predictictors, create interaction terms
# plot vs outcome to see if any worth retaining

quadInts <- quadInteraction(numCols)
quadIntsT <- quadInteraction(numColsT)

# density plots for double
colNames <- names(quadInts)
j <- 1
plotListInts <- list()
for(i in colNames){
  plt <- ggplot(quadInts, aes_string(x=i)) +
    geom_density(aes(color = trainOutcome)) +
    geom_rug() +
    scale_colour_brewer(palette = "Set1") +
    theme_bw()
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListInts[[i]] <- plt
}

multiplotList(plotListInts[1:12], cols = 3)

# two options- cherry pick the interaction terms that seperate well,
# or throw all into model. Start by including all
# calculate overlaps of denisty functions??? That would be an ace criteria for
# retaining terms

# try transforming as well, as some degenerate
# DOESNT REALLY SEEM TO HELP. May show that linear models
# not suitable, as still some degenerate

trans <- preProcess(quadInts, method = "BoxCox")
quadIntTrans <- predict(trans, quadInts)
quadIntTransT <- predict(trans, quadIntsT)

colNames <- names(quadIntTrans)
j <- 1
plotListquadIntTrans <- list()
for(i in colNames){
  plt <- ggplot(quadIntTrans, aes_string(x=i)) +
    geom_density(aes(color = trainOutcome)) +
    geom_rug() +
    scale_colour_brewer(palette = "Set1") +
    theme_bw()
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListquadIntTrans[[i]] <- plt
}

multiplotList(plotListquadIntTrans[1:12], cols = 3)

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
# for number vmail, just have binary yes no for zero or not zero
# thers, take sqrt or log to resolve skewness
intColsTrans <- intCols %>%
  mutate(number_vmail_messages = ifelse(number_vmail_messages == 0, 0, 1),
         number_customer_service_calls = sqrt(number_customer_service_calls),
         total_intl_calls = log(total_intl_calls + 1))

intColsTransT <- intColsT %>%
  mutate(number_vmail_messages = ifelse(number_vmail_messages == 0, 0, 1),
         number_customer_service_calls = sqrt(number_customer_service_calls),
         total_intl_calls = log(total_intl_calls + 1))

colNames <- names(intColsTrans)
j <- 1
plotListIntTrans <- list()
for(i in colNames){
  plt <- ggplot(intColsTrans, aes_string(x = i)) +
    scale_colour_brewer(palette = "Set1") +
    geom_histogram(aes(color = trainOutcome), alpha = 0.4) +
    theme_bw()
  
  assign(paste("plot", j, sep = ""), plt)
  j <- j + 1
  plotListIntTrans[[i]] <- plt
}

multiplotList(plotListIntTrans, cols = 3)

##################################################
# step 5: combine predictors
# do not do transformed at this stage
# not thet factTrans is just transformed to dummy
trainInput <- cbind(quadIntTrans, facTrans, intColsTrans)
testInput <- cbind(quadIntTransT, facTransT, intColsTransT)

# step 5b: same processing for test set trainInputT

# step 6: make full and reduced set

# full set: remove ZV
# reduced set: remove nzv
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

# glm
set.seed(476)
glmTune <- train(x = trainInput[, reducedSet],
                 y = trainOutcome,
                 method = "glm",
                 preProc = c("center", "scale"),
                 metric = "Sens",
                 trControl = ctrl)

glmTune

glmPred <- predict(glmTune, newdata = testInput[, reducedSet])
glmPredProb <- predict(glmTune, newdata = testInput[, reducedSet], type = "prob")
confusionMatrix(data = glmPred, reference = testOutcome)

glmPredProb$class <- testOutcome

calCurve <- calibration(class ~ yes, data = glmPredProb)
xyplot(calCurve,
       auto.key = list(columns = 2))

liftCurve <- lift(class ~ yes, data = glmPredProb)
liftCurve

xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = T,
                       points = F))

# lda
set.seed(476)
ldaTune <- train(x = trainInput[, reducedSet],
                 y = trainOutcome,
                 method = "lda",
                 preProc = c("center", "scale"),
                 metric = "Sens",
                 trControl = ctrl)

ldaTune

# plsda
set.seed(476)
plsdaTune <- train(x = trainInput[, reducedSet],
                   y = trainOutcome,
                   method = "pls",
                   tuneGrid = expand.grid(ncomp = 1:10),
                   preProc = c("center", "scale"),
                   probMethod = "Bayes",
                   metric = "Sens",
                   trControl = ctrl)

plsdaTune
plot(plsdaTune)

# glmnet - can use full set
set.seed(476)
glmnetGrid <- expand.grid(alpha = c(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1),
                          lambda = seq(0.0, 0.38, length = 20))
glmnetTune <- train(x = trainInput[, fullSet],
                    y = trainOutcome,
                    method = "glmnet",
                    tuneGrid = glmnetGrid,
                    preProc = c("center", "scale"),
                    metric = "Sens",
                    family = "binomial",
                    trControl = ctrl)

glmnetTune

# penalised lda
set.seed(476)
penldaGrid <- expand.grid(NumVars = seq(0, 20),
                          lambda = c(0.05, 0.1, 0.15, 0.2))
penalisedldaTune <- train(x = trainInput[, fullSet],
                          y = trainOutcome,
                          method = "sparseLDA",
                          tuneGrid = penldaGrid,
                          preProc = c("center", "scale"),
                          metric = "Sens",
                          trControl = ctrl)

penalisedldaTune

# nsc - can use full set
nscGrid <- data.frame(threshold = 0:20)
set.seed(476)
nscTune <- train(x = trainInput[, fullSet],
                  y = trainOutcome,
                  method = "pam",
                  tuneGrid = nscGrid,
                  preProcess = c("center", "scale"),
                  metric = "Sens",
                  trControl = ctrl)

nscTuned

# step 8: predictions.

glmPred <- predict(glmTune, newdata = testInput[, reducedSet])
glmPredProb <- predict(glmTune, newdata = testInput[, reducedSet], type = "prob")
confusionMatrix(data = glmPred, reference = testOutcome)

ldaPred <- predict(ldaTune, newdata = testInput[, reducedSet])
ldaPredProb <- predict(ldaTune, newdata = testInput[, reducedSet], type = "prob")
confusionMatrix(data = ldaPred, reference = testOutcome)

plsdaPred <- predict(plsdaTune, newdata = testInput[, reducedSet])
plsdaPredProb <- predict(plsdaTune, newdata = testInput[, reducedSet], type = "prob")
confusionMatrix(data = plsdaPred, reference = testOutcome)

glmnetPred <- predict(glmnetTune, newdata = testInput[, fullSet])
glmnetPredProb <- predict(glmnetTune, newdata = testInput[, fullSet], type = "prob")
confusionMatrix(data = glmnetPred, reference = testOutcome)

penalisedldaPred <- predict(penalisedldaTune, newdata = testInput[, fullSet])
penalisedldaPredProb <- predict(penalisedldaTune, newdata = testInput[, fullSet], type = "prob")
confusionMatrix(data = penalisedldaPred, reference = testOutcome)

nscPred <- predict(nscTune, newdata = testInput[, fullSet])
nscPredProb <- predict(nscTune, newdata = testInput[, fullSet], type = "prob")
confusionMatrix(data = nscPred, reference = testOutcome)

# step 9: calibration and lift plot
results <- data.frame(glm = glmPredProb$yes,
                      lda = ldaPredProb$yes,
                      plsda = plsdaPredProb$yes,
                      glmnet = glmnetPredProb$yes,
                      penalisedlda = penalisedldaPredProb$yes,
                      nsc = nscPredProb$yes,
                      class = testOutcome)


calCurve <- calibration(class ~ glm + lda + plsda + glmnet + penalisedlda + nsc, data = results)
calCurve
xyplot(calCurve,
       auto.key = list(columns = 3))

liftCurve <- lift(class ~ glm + lda + plsda + glmnet + penalisedlda + nsc, data = results)
liftCurve

xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = T,
                       points = F))

# step 10: recalibrate probabilities

glmBayes <- NaiveBayes(class ~ glm, data = results, usekernel = TRUE)
glmBayesProbs <- predict(glmBayes, newdata = results[, "glm", drop = F])
results$glmBayes <- glmBayesProbs$posterior[, "yes"]
confusionMatrix(data = glmBayesProbs$class, reference = testOutcome)

ldaBayes <- NaiveBayes(class ~ lda, data = results, usekernel = TRUE)
ldaBayesProbs <- predict(ldaBayes, newdata = results[, "lda", drop = F])
results$ldaBayes <- ldaBayesProbs$posterior[, "yes"]
confusionMatrix(data = ldaBayesProbs$class, reference = testOutcome)

plsdaBayes <- NaiveBayes(class ~ plsda, data = results, usekernel = TRUE)
plsdaBayesProbs <- predict(plsdaBayes, newdata = results[, "plsda", drop = F])
results$plsdaBayes <- plsdaBayesProbs$posterior[, "yes"]
confusionMatrix(data = plsdaBayesProbs$class, reference = testOutcome)

glmnetBayes <- NaiveBayes(class ~ glmnet, data = results, usekernel = TRUE)
glmnetBayesProbs <- predict(glmnetBayes, newdata = results[, "glmnet", drop = F])
results$glmnetBayes <- glmnetBayesProbs$posterior[, "yes"]
confusionMatrix(data = glmnetBayesProbs$class, reference = testOutcome)

penldaBayes <- NaiveBayes(class ~ penalisedlda, data = results, usekernel = TRUE)
penldaBayesProbs <- predict(penldaBayes, newdata = results[, "penalisedlda", drop = F])
results$penldaBayes <- penldaBayesProbs$posterior[, "yes"]
confusionMatrix(data = penldaBayesProbs$class, reference = testOutcome)

nscBayes <- NaiveBayes(class ~ nsc, data = results, usekernel = TRUE)
nscBayesProbs <- predict(penldaBayes, newdata = results[, "nsc", drop = F])
results$nscBayes <- nscBayesProbs$posterior[, "yes"]
confusionMatrix(data = nscBayesProbs$class, reference = testOutcome)


calCurve2 <- calibration(class ~ glmBayes + ldaBayes + plsdaBayes + 
                           glmnetBayes + penldaBayes + nscBayes, data = results)
xyplot(calCurve2,
       auto.key = list(columns = 3))

liftCurve2 <- lift(class ~ glmBayes + ldaBayes + plsdaBayes + 
                     glmnetBayes + penldaBayes + nscBayes, data = results)
liftCurve2

xyplot(liftCurve2,
       auto.key = list(columns = 2,
                       lines = T,
                       points = F))

# make a funky plot extracting ROC and kappa for test set
# (dot plots) - before and after bayes
# lots of repeated junk here- tidy up if ever think about showing to the world
Kappas <- data.frame(kappa = c(confusionMatrix(data = glmPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = ldaPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = plsdaPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = glmnetPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = penalisedldaPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = nscPred, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = glmBayesProbs$class, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = ldaBayesProbs$class, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = plsdaBayesProbs$class, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = glmnetBayesProbs$class, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = penldaBayesProbs$class, reference = testOutcome)$overall[[2]],
                                confusionMatrix(data = nscBayesProbs$class, reference = testOutcome)$overall[[2]]),
                     model = rep(c("glm", "lda", "plsda", "glmnet", "penlda", "nsc"), 2),
                     type = rep(c("No Cal", "Cal"), each = 6) )

plot1 <- ggplot(Kappas, aes(x = kappa,
                                 y = reorder(model, kappa),
                                 color = type,
                                 shape = type) ) +
  geom_point(alpha = 0.6, size = 5) +
  scale_shape_manual(values=c(16, 17)) +
  scale_colour_brewer(palette="Set1") +
  scale_x_continuous(limits = c(0.1, 0.5)) +
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


# AUC for ROC
# bare in mind that tuned models for Sens, not ROC or kappa... 

rocCurves <- lapply(results[,!names(results)=="class"],
                    function(x) roc(response = testOutcome,
                                    predictor = x,
                                    levels = rev(levels(results$class))))

aucROC <- lapply(rocCurves, function(x) auc(x)[[1]])

dfAUC <- do.call(rbind.data.frame, aucROC)
dfAUC$model <- row.names(dfAUC)
dfAUC$type <- rep(c("No Cal", "Cal"), each = 6)

dfAUC$model <- gsub("Bayes", "", dfAUC$model)
dfAUC$model <- gsub("penalisedlda", "penlda", dfAUC$model)
names(dfAUC)[1] <- "ROC"
rownames(dfAUC) <- NULL

plot2 <- ggplot(dfAUC, aes(x = ROC,
                            y = reorder(model, ROC),
                            color = type,
                            shape = type) ) +
  geom_point(alpha = 0.6, size = 5) +
  scale_shape_manual(values=c(16, 17)) +
  scale_colour_brewer(palette="Set1") +
  scale_x_continuous(limits = c(0.7, 0.9)) +
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


# best model, 80% of churns
# best is penlda, no cal
penldaLift <- (liftCurve$data[liftCurve$data$liftModelVar=="penalisedlda", ])
plot( penldaLift$CumEventPct  ~ penldaLift$CumTestedPct , type = 'l')
penldaLift[penldaLift$CumEventPct <80.1 & penldaLift$CumEventPct > 79.9,  ]

# so approx 35% of all samples required to find 80% of churns
# = 0.35 * 1667

table(trainOutcome)
# there are 483 churns
# 0.8 * 483 = 386

# 0.35 * 1667 - 386 = 197
# so 197 non churns would be identified with best model
# to find 80% of churners (386)

# ?? not sure. think about 

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