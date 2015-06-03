library(car)
library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(e1071)
library(mlbench)
library(reshape2)
library(vcd)

data(Glass)
str(Glass)

# lattice plots are good for exploratory
# first melt into long
meltedGlass <- melt(Glass, id.vars = 'Type')

densityplot(~value | variable,
            data= meltedGlass,
            # adjust axis so measure scale different for each panel
            scales = list(x = list(relation = 'free'),
                          y = list(relation = 'free')),
            # adjust smooths
            adjust = 1.25,
            # symbol on the rug for each data point
            pch = '|',
            xlab = 'Predictor')

# if do not like lattice, use ggplot2. Relies on my modified
# multiplot function
colNames <- names(Glass)[1:9]
j <-1
plotList <- list()
for(i in colNames){
  plt <- ggplot(Glass, aes_string(x=i)) + geom_density() + geom_rug()
  assign(paste("plot", j, sep = ""), plt)   
  j <- j+1
  plotList[[i]] <- plt
}
multiplotList(plotList[1:9],cols=3)

# make scatterplot matrix using lattice
splom(~Glass[, 1:9], pch = 16, cex = 0.7 )

# or, if not too familier with lattice use a scatterplotMatrix 
scatterplotMatrix(Glass[, 1:9], diagonal='none', smoother='none', reg.line='none')

# what transforms would help?
# Box-Cox cannot be applied as have many zero values
# use Yeo-Johnson instead!

yjTrans <- preProcess(Glass[, 1:9], method = 'YeoJohnson')
yjData <- predict(yjTrans, newdata = Glass[, 1:9])


# we see that it has not helped hugely...

splom(~yjData[, 1:9], pch = 16, cex = 0.7 )

colNames <- names(yjData)[1:9]
j <-1
plotList <- list()
for(i in colNames){
  plt <- ggplot(yjData, aes_string(x=i)) + geom_density() + geom_rug()
  assign(paste("plot", j, sep = ""), plt)   
  j <- j+1
  plotList[[i]] <- plt
}
multiplotList(plotList[1:9],cols=3)

# try removing outliers

centerScale <- preProcess(Glass[, 1:9], method = c('center', 'scale'))
csData <- predict(centerScale, newdata = Glass[, 1:9])
ssData <- spatialSign(csData)
splom(~ssData, pch = 16, cex = 0.7)

# overall, we were unable to resolve skewness with transforms, so tree based
# models would be most appropriate



#------------------------------------------------------------------

multiplotList <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <-  c(..., plotlist)
  
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
