library(caret)
library(AppliedPredictiveModeling)
library(reshape2)
data(ChemicalManufacturingProcess)

str(ChemicalManufacturingProcess)

# just for fun
chemMelt <- melt(ChemicalManufacturingProcess)
densityplot(~value|variable,
            data = chemMelt,
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")),
            adjust = 1.25,
            pch = "|",
            xlab = "predictor")

# consult the table given in the question

# using 'one standard error' method, we see that the best mean R^2 is 0.545, with error 0.0308
# 0.545 - 0.0308 = 0.5142, therefore the model wth 2 components is the most parsimonious model

# tolerance values are given from (X-O)/O where X is performance value, and O is the numerically
# optimal value. if 10% loss in R^2 acceptable we have -0.1 = (X - 0.545) / 0.545 = 0.4905
# so optimum number of components is 2

# consulting the figure, would cleary chose Random forest or SVM- by eye (and no doubt using t-test
# or diff() ) there wuold be no statistically significant difference between the two results

# random forest is most acccurate, but slowest and not too interpretable.
# SVM signifcantly faster, but once again may suffer from interpretability issues
# regression tree and pls are fast, interpretable and comparitively accurate

set.seed(19711230)
plsProfileChemMod <- train(Yield ~ .,
                           data = ChemicalManufacturingProcess,
                           method = 'pls',
                           preProc = c('center', 'scale'),
                           tuneLength = 10,
                           trControl = trainControl(method = 'repeatedcv', repeats = 5))

R2values <- plsProfileChemMod$results[, c('ncomp', 'Rsquared', 'RsquaredSD')]
# this gets standard error in the mean as did resampling - need size of the samples
R2values$RsquaredSEM <- R2values$RsquaredSD/sqrt(length(plsProfileChemMod$control$index))

oneSE <- ggplot(R2values, aes(ncomp, Rsquared, ymin = Rsquared - RsquaredSEM,
                              ymax = Rsquared))
oneSE + geom_linerange() + geom_pointrange() + theme_bw()

# one standard error rule
bestR2 <- subset(R2values, ncomp == which.max(R2values$Rsquared))
bestR2$lb <- bestR2$Rsquared  - bestR2$RsquaredSEM
candR2 <- subset(R2values, Rsquared >= bestR2$lb & ncomp < bestR2$ncomp )

# tolerance rule

bestR2 <- subset(R2values, ncomp == which.max(R2values$Rsquared))
R2values$tolerance <- (R2values$Rsquared - bestR2$Rsquared) / bestR2$Rsquared * 100

# optimim number of components is 2 (agrees with other answer)