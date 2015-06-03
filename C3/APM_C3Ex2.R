library(car)
library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(e1071)
library(mlbench)
library(reshape2)
library(vcd)

data(Soybean)
str(Soybean)

# Soybean is mainly factors, which makes it most awkward!
# many of the factors coded not to helpfully
# use recode() in car package

Soybean2 <- Soybean

table(Soybean2$temp, useNA = 'always')

Soybean2$temp <- recode(Soybean2$temp,
                        "0 = 'low'; 1 = 'norm'; 2 = 'high'; NA = 'missing'",
                        levels = c('low', 'norm', 'high', 'missing'))
# much better
table(Soybean2$temp)

# recode date and precip as well
Soybean2$date <- recode(Soybean2$date,
                        "0 ='apr';1='may';2='june';3='july';4='aug';5='sept';6='oct';NA = 'missing'",
                        levels = c("apr", "may", "june", "july", "aug", "sept", "missing"))
 
Soybean2$precip <- recode(Soybean2$precip,
                          "0 = 'low'; 1 = 'medium'; 2 = 'high'; NA = 'missing'",
                          levels = c('low', 'medium', 'high', 'missing'))

# make a funky mosaic plot!
mosaic(~date + temp , data = Soybean2)

# or bar chart. Higher dimensions arr easier to visulise
barchart(table(Soybean2$date, Soybean2$temp, Soybean2$precip),
         auto.key = list(columns = 4, title = 'temperature'))

# lets have a look at each Class in the dataset, and see how many 
# complete cases we have (that is, no missing values)
# we see that some may be problematic!

table(Soybean$Class, complete.cases(Soybean))

# which predictors have missing values??
hasMissing <- unlist(lapply(Soybean, function(x) any(is.na(x))))
head(hasMissing)
hasMissing <- names(hasMissing)[hasMissing]
head(hasMissing)

# there are several classes where all of the predictors have at least one missing value
# are these concentrated in single predictor that could be removed?

# get the percentage of missing values for each predictor by class with the syntax:
byPredByClass <- apply(Soybean[, hasMissing], 2,
                       function(x, y) {
                         tab <- table(is.na(x), y)
                         tab[2, ]/apply(tab, 2, sum)
                       },
                       y = Soybean$Class)

# colmuns are predictors and the rows are classes. Eliminate those with
# no missing values

byPredByClass <- byPredByClass[apply(byPredByClass, 1, sum) > 0, ]
byPredByClass <- byPredByClass[, apply(byPredByClass, 2, sum) > 0]

t(byPredByClass)

# we have options then. Could impute? But unlikely to help as nearly all predictors
# are missing for some cases.
# could just remove those without complete cases
# could make NA a class itself (recode so 'missing' is a class)

# how will frequencies affect model? Convert to dummies and see if some have
# near zero variance.

orderedVars <- unlist(lapply(Soybean, is.ordered))
orderedVars <- names(orderedVars)[orderedVars]

# just remove the offending classes with missing data

completeClasses <- as.character(unique(Soybean$Class[complete.cases(Soybean)]))
Soybean3 <- subset(Soybean, Class %in% completeClasses)
for(i in orderedVars) Soybean3[, i] <- factor(as.character(Soybean3[, i]))

dummyInfo <- dummyVars(Class ~ ., data=Soybean3)
dummies <- predict(dummyInfo, Soybean3)

predDistInfo <- nearZeroVar(dummies, saveMetrics = TRUE)
sum(predDistInfo$nzv)

# want to retain those without nzv
keep <- dummies[, predDistInfo$nzv]