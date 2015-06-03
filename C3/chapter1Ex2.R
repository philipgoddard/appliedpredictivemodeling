library(mlbench)
library(AppliedPredictiveModeling)
library(caret); library(corrplot)
library(e1071); library(lattice); library(dplyr)

data(Soybean)

str(Soybean)

#can get summary of data
summary(Soybean)

#investigate individual predictors with tables
table(Soybean$seed.size)

#are any degenerate (ie have the same information)
#plot various ones
plot(Soybean$date)
plot(Soybean$plant.stand)
plot(Soybean$precip)

#from summary and more plots, it seems several seem degenerate
# for example mold.growth and seed.discolor

#better way to test? chisq?

tbl = matrix(data=c(524, 513, 67, 64, 92, 106), nrow=3, ncol=2, byrow=T)
dimnames(tbl) = list(level=c('0', '1', 'NA'), pred=c('mold.growth', 'mold.discolor'))

chi2 = chisq.test(tbl, correct=F)
c(chi2$statistic, chi2$p.value)

#chi sq p value is quite large- so would reject hyptothesis
#of independence. correlation is 0.55

#invetsigate the checkResamples() and checkConditionalX() functions
#need dummy variables... check ?nearZeroVar


#how many NA in each predictor?
apply(Soybean, 2, function(x) {sum(is.na(x))})

#we see some pattern in missing data- for example if seed is
#na, so is seed.size

#handling missing data - can we impute categorical?
#do we have to first make dummy variables, then impute and round?

dummies <- dummyVars(Class ~ ., data=Soybean)
soyDummy <- predict(dummies, Soybean)

#NO- a little research on google shows this is bad method
#just deleting row is better method than this! but do get bias...
#other ways like logistic regression can work


