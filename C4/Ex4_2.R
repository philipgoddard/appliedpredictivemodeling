library(caret)
library(AppliedPredictiveModeling)
data(permeability)

str(permeability)
hist(permeability)

# number of samples is small (165)
# would have to focus upon approach oteh rthan splitting into training and testing

# cross validation is therefore sensible

# very skewed- most samples have low value

set.seed(123)

repeatedCV <- createMultiFolds(permeability, k = 10, times = 25)

# can have a look at the folds

hist(permeability[repeatedCV$Fold01.Rep01, ])
hist(permeability[repeatedCV$Fold02.Rep01, ])
hist(permeability[repeatedCV$Fold03.Rep01, ])
hist(permeability[repeatedCV$Fold04.Rep01, ])

# first fold 
cv1 <- permeability[repeatedCV$Fold01.Rep01]

# to apply in actual models, use 
# trControl=trainControl(method = "25", number = 10, repeats = 25)
# when using train()