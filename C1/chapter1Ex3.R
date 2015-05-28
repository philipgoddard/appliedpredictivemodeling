library(mlbench)
library(AppliedPredictiveModeling)
library(caret); library(corrplot)
library(e1071); library(lattice); library(dplyr)
library(ggplot2)

data(BloodBrain)

summary(bbbDescr)
sum(is.na(bbbDescr))

head(logBBB) #outcome

#do any of the individual predictors have degenerate 
#distributions? have a quick plot. and the answer is YES
qplot(bbbDescr[,5], geom='density')

#look for correlated predictors
correlation <- cor(bbbDescr)
corrplot(correlation, order = "hclust")
#looks like lots correlated

ncol(bbbDescr)
highCorr <- findCorrelation(correlation, cutoff = .75)
filteredbbbDescr <- bbbDescr[, -highCorr]
ncol(filteredbbbDescr)

#down from 134 to 68 predictors by removing those correlated
#above threshold
