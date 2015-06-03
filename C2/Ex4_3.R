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