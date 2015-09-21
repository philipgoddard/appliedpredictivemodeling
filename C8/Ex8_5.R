# 8.5

# Fit different tree or rule-based models for the Tecator data

library("caret")
library("randomForest")
library("rpart")
library("Cubist")
library("doMC")
library("e1071")
library("gbm")
library("RWeka")

data(tecator)
registerDoMC(cores = 4)

# absord is a matrix, contain 100 absorbance values for the 215 samples
head(absorp)
summary(absorp)
# endpoints is a matrix containing percentage moisture, fat and protein in
# columns 1-3 respectively
head(endpoints)
summary(endpoints)
# we are predicting upon fat, so this is our target
fat <- endpoints[, 2]
# check distribution, pretty skewed
skewness(fat)
# take sqrt fro much less skewed outcome
skewness(sqrt(fat))

fatTrans <- sqrt(fat)

# just have to remember to transform back
# when reporting predictions!

inTrain <- createDataPartition(fatTrans,
                               p=0.75,
                               list = FALSE)

set.seed(100)
# train test split
trainingAbsorp <- absorp[inTrain, ]
testingAbsorp <- absorp[-inTrain, ]
trainingFat <- fatTrans[inTrain]
testingFat <- fatTrans[-inTrain]

# set up cross validation method - use repeated CV 
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)

# recursive partition tree
# tuning over complexity param here
set.seed(100)
rpartTune <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
                   method = "rpart2",
                   tuneLength = 10,
                   trControl = ctrl)

# try to bag
# have to make custom functions to do aggregate diffrent
# types of model to do this - 
# there are several in caret bag() functon to do this

# lets take the easy route and just use built in CART bag
set.seed(100)
rpartBag <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
                  method = "treebag",
                  trControl = ctrl)


# rule based tree
# too slow for now

 #apparantly no like parallel
#set.seed(100)
#detach("package:doMC", unload = T)
#set.seed(100)
#m5Tune <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
#                method = "M5",
                # Use an option from M5() to specify the minimum
                # number of samples needed to further split the 
                # data to be 10
#                control = Weka_control(M = 10),
#                trControl = ctrl)
#
#m5Tune


#library("doMC")
#registerDoMC(cores = 4)


# random forest
set.seed(100)
rForestTune <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
                     method = "rf",
                     verbose = FALSE,
                     ntree = 1000,
                     trControl = ctrl)

# boosted tree
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(500, 1000, by = 100),
                       shrinkage = c(0.01, 0.1),
                       n.minobsinnode = 10)
set.seed(100)
gbmTune <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 verbose = FALSE)

gbmTune

# cubist
# apparantly need to put data frame object in here. 
set.seed(100)
cubistTune <- train(x = as.data.frame(trainingAbsorp), y = trainingFat,
                      method = "cubist",
                      verbose = FALSE,
                      tuneGrid = expand.grid(committees = c(1:10, 20, 50, 100),
                                               neighbors = c(0, 1, 5, 9)),
                      trControl = ctrl)

cubistTune


# now do comparisons of models, analysis of tuning params etc etc etc

