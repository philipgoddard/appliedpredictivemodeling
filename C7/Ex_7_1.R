library(caret)
library(kernlab)
library(reshape2)


set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)
plot(x, y)

## Create a grid of x values to use for prediction
dataGrid <- data.frame(x = seq(2, 10, length = 100))

rbfSVM <- ksvm(x = x, y = y, data = sinData,
               kernel ="rbfdot", kpar = "automatic",
               C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)

## This is a matrix with one column. We can plot the
## model predictions by adding points to the previous plot

points(x = dataGrid$x, y = modelPrediction[,1],
       type = "l", col = "blue")

## investigate different values of cost C and
## epsilon
epsilonValues <- seq(0.1, 0.5, 0.05)
CValues = unlist(lapply(seq(-2,11), function(x) 2^x))


plot(x, y)
# for efficieny should decalre pred to be a matrix of defined size
pred <- NULL

for (i in seq_along(epsilonValues)) {
  for (j in seq_along(CValues)){
    
    set.seed(100)
    rbfSVM <- ksvm(x = x, y = y, data = sinData,
                   kernel ="rbfdot", kpar = "automatic",
                   C = CValues[j], epsilon = epsilonValues[i])
    
    modelPrediction <- predict(rbfSVM, newdata = dataGrid)
    pred <- cbind(pred, modelPrediction)
    
    tmp <- data.frame(x = dataGrid$x,
                      y = modelPrediction,
                      C = CValues[j],
                      epsilon = epsilonValues[i])
    
    results <- if(i ==1 && j == 1) tmp else rbind(results, tmp)
    
    tmp2 <- data.frame(C = CValues[j],
                       epsilon = epsilonValues[i],
                       rmse = RMSE(modelPrediction[, 1], sinData$y))

    rmseResults <- if(i == 1 && j == 1) tmp2 else rbind(rmseResults, tmp2)
  }
  # plot over each other results
  points(x = dataGrid$x, y= pred[, i], type = 'l', col = 'red')
}

# do a classy xy plot (inspired by solutions!)
xyplot(y~x|as.factor(C)*as.factor(epsilon), 
       data = results, 
       panel = function(...) {
         panel.xyplot(sinData$x, sinData$y,
                      col = rgb(.2, .2, .2, .3),
                      pch = 16, cex = .7)
         panel.xyplot(...)
       },
       ylim = extendrange(sinData$y),
       type= "l", lwd = 2)


# generally speaking, model becomes more jagged as cost increases.
# this is as training process puts cost on residual values
# higher cost will tell model to reduce residual by overfitting
# this gives LOW bias and HIGH variability (eg the model will change if
# single data points change)
# as epsilon decreases model overfits more, but effect is not as potent as cost.


# add in a loop for sigma...

# use as.vector(sigest(y~x, data = sinData, frac = 0.75))
# to get values of estimated sigma to use

