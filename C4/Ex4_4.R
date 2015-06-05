library(caret)

data(oil)
str(oilType)
qplot(oilType)
table(oilType)

str(fattyAcids)

# use sample() in base R to create a completely random sample
# of  60 oils

table(sample(oilType, 60, replace = FALSE))
qplot(sample(oilType, 60, replace = FALSE))

# see lots of variation - stratified sampling is very important!
# especially as some classes have few 

oilSampStratified <- createDataPartition(oilType,
                               p=0.6,
                               list = FALSE)

# Pull out those in the statified sample andplot
qplot(oilType[oilSampStratified])

# we see that it is much more similar to unsampled distribution when 
# using createDataPartition

# With a small sample, cross validation is essential to determine performance of model
# 10-fold cross validation is recommended, as the bias and variance properties are good,
# and the computational costs not too large

# Good way to understand uncertainty of test set is confidence interval.
# can use binom.test() 
# Input the number of samples and number correctly classified in test set
# for example if 20 in test set, and classify 16 correctly:
binom.test(16, 20)
binom.test(16, 20)$conf.int[2] - binom.test(16, 20)$conf.int[1]

# We see that the width of the 95% confidence interval is 37.9%
# we see as the number correct gets larger, the confidence interval gets smaller

#-----------------------------
# Textbook answers:

# a) random sample
sampNum <- floor(length(oilType) * 0.6) + 1
set.seed(629)
oilSplits <- vector(mode = "list", length = 20)

for(i in seq(along = oilSplits)){oilSplits[[i]] <- table(sample(oilType, size = sampNum))}

head(oilSplits, 3)

oilSplits <- do.call("rbind", oilSplits)
head(oilSplits, 3)

summary(oilSplits / sampNum)

# b) stratified sample
set.seed(629)
oilSplits2 <- createDataPartition(oilType, p = 0.6, times = 20)
oilSplits2 <- lapply(oilSplits2, function(x, y) table(y[x]), y = oilType )

head(oilSplits2)
oilSplits2 <- do.call("rbind", oilSplits2)
summary(oilSplits2/sampNum)

# difficult to choose a data splitting strategy. Once possibility is LOOCV
# with the exception of G, each class will be represented in at least one resample.
# Some classification models, however, require at least one sample of each class.
# It may be reasonably to rely only on leave-one-out CV to assess performance
# Alternatively, a test set consisting of only classes with most samples could be used (A,B,E,F)

# step 1 - split for train test (if appropriate)
# step 2 - chose data splitting for cross validation (when fitting model) - get in sample error,
# step 3 - (if applicable) test on testing data get out of sample error
