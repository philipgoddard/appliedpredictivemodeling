# 8.1

# NOTE that nice varimp plots only work
# for objects of class "varImp.train"
# eg plot(varImp(blah))

library("caret")
library("randomForest")
library("party")
library("mlbench")
library("doMC")

registerDoMC(cores = 4)

set.seed(200)
simulated <- mlbench.friedman1(200, sd = 1)
simulated <- cbind(simulated$x, simulated$y)
simulated <- as.data.frame(simulated)
colnames(simulated)[ncol(simulated)] <- "y"

# fit a random forest to data
model1 <- randomForest(y ~ ., data = simulated,
                       importance = TRUE,
                       ntree = 1000)
rfImp1 <- varImp(model1, scale = FALSE)

# we see that the random forest did
# not significantly use uninformatve predictors (V6-V10)

# add another predictor that is significantly correlated
# with another predictor

simulated$duplicate1 <- simulated$V1 + rnorm(200) * .1
cor(simulated$duplicate1, simulated$V1)

model2 <- randomForest(y ~ ., data = simulated,
                       importance = TRUE,
                       ntree = 1000)
rfImp2 <- varImp(model2, scale = FALSE)

# we see that the importance of V1 is diluted somewhat
# and that duplicate is relatively important

# add another predictor
simulated$duplicate2 <- simulated$V1 + rnorm(200) * .1

model3 <- randomForest(y ~ .,
                       data = simulated,
                       importance = TRUE,
                       ntree = 1000)
rfImp3 <- varImp(model3, scale = FALSE)

# see further dilution

# Use the cforest function in the party package to fit a random forest model
# using conditional inference trees.
bagCtrl <- cforest_control(ntree = 1000)
model4 <- cforest(y ~ .,
                  data = simulated,
                  control = bagCtrl)

varimp(model4, conditional = F)
# conditional = T seems very slow
varimp(model4, conditional = T)

# we see similar results for conditional = F to random forest

library("Cubist")
model5 <- cubist(x = simulated[, names(simulated) != "y"],
                 y = simulated$y,
                 committees = 100)

cubistImp <- varImp(model5)

# odd, as duplicate 2 way more important than duplicate 1

# probably should repeat fitting conditional forests and cubist to 
# all three sets of data to compare effect as gradually build up

