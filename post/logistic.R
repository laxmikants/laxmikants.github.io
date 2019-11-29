
```{r}
plotData <-
  function (X, y, axLables = c("#studied","#slept"), legLabels =
              c('passed', 'Not passed')) {
  
    
    symbolss <- c(3,21) #plus and empty circle character codes
    
    ############### This part is for legend to be plotted above the plot
    plot(X[,1],X[,2],type = "n", xaxt = "n", yaxt = "n")
    leg <- legend(
      "topright",legLabels, pch = rev(symbolss),
      pt.bg = "yellow", plot = FALSE
    )
    
    #custom ylim. Add the height of legend to upper bound of the range
    yrange <- range(X[,2])
    yrange[2] <- 1.04 * (yrange[2] + leg$rect$h)
    ###############
    
    yfac <- factor(y)
    plot(
      X[,1],X[,2], pch = symbolss[yfac] ,bg = "yellow", lwd = 1.3,
      xlab = axLables[1], ylab = axLables[2],
      ylim = yrange
    )
    
    legend("topright",legLabels,pch = rev(symbolss),
           pt.bg = "yellow")
    # ----------------------------------------------------
  }
```

```{r}
plotDecisionBoundary <-
  function (theta, X, y, axLables = c("#Studied","#Slept"), legLabels =
              c('Passed', 'Not Passed')) {
    
    # Plot Data
    plotData(X[,2:3], y,axLables,legLabels)
    
    if (dim(X)[2] <= 3)
    {
      # Only need 2 points to define a line, so choose two end points
      plot_x <- cbind(min(X[,2] - 2), max(X[,2] + 2))
      # Calculate the decision boundary line
      plot_y <- -1 / theta[3] * (theta[2] * plot_x + theta[1])
      
      # Plot, and adjust axes for better viewing
      lines(plot_x, plot_y, col = "blue")
      
    }
    else
    {
      # Here is the grid range
      u <- seq(-1,1.5, length.out = 50)
      v <- seq(-1,1.5, length.out = 50)
      
      z <- matrix(0, length(u), length(v))
      # Evaluate z <- theta*x over the grid
      for (i in 1:length(u))
        for (j in 1:length(v))
          z[i,j] <- mapFeature(u[i], v[j]) %*% theta
      
      # Notice you need to specify the range [0, 0]
      contour(
        u, v, z, xlab = 'Microchip Test 1', ylab = 'Microchip Test 2',
        levels = 0, lwd = 2, add = TRUE, drawlabels = FALSE, col = "green"
      )
      mtext(paste("lambda = ",lambda), 3)
    }
  }
```


```{r}
predict <- function(theta, X) {
  
  m <- dim(X)[1] # Number of training examples
  
  p <- rep(0,m)
  
  p[sigmoid(X %*% theta) >= 0.5] <- 1
  
  p
  # ----------------------------------------------------
}
```


```{r}
studied <- c(4.85,8.62,5.43,9.21)

slept <- c(9.63,3.23,8.23,6.34)

y    <- c(1, 0, 1, 0)

X <- data.frame(studied,slept)

X <- as.matrix(X)

```

```{r}
plotData(X,y)
```


```{r}
m <- dim(X)[1]

n <- dim(X)[2]


# Add intercept term to x and X_test
X <- cbind(rep(1,m),X)

# Initialize fitting parameters
initial_theta <- rep(0,n+1)

cF <- costFunction(X, y)(initial_theta)

cost <- costFunction(X, y)(initial_theta)

grd <- grad(X,y)(initial_theta)

```

```{r}
optimRes <- optim(par = initial_theta, fn = costFunction(X,y), gr = grad(X,y), 
                  method="BFGS", control = list(maxit = 400))
theta <- optimRes$par
cost <- optimRes$value


```


```{r}
# Print theta to screen
cat(sprintf('Cost at theta found by optim: %f\n', cost))
cat(sprintf('theta: \n'))
cat(sprintf(' %f \n', theta))
```

```{r}
# Plot Boundary
plotDecisionBoundary(theta, X, y)
```

```{r}
prob <- sigmoid(t(c(1,8,6)) %*% theta)
cat(sprintf('For a student study hrs5 and sleep hrs 4, we predict an pass probability of\n %f\n', prob))
# Compute accuracy on our training set
p <- predict(theta, X)
cat(sprintf('Train Accuracy: %f\n', mean(p == y) * 100))

```


```{r}
Hours <- c(0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25,
           2.50, 2.75, 3.00, 3.25, 3.50,    4.00,   4.25,   4.50,   4.75,
           5.00, 5.50)


Pass    <- c(0, 0, 0, 0, 0, 0, 1,   0, 1, 0, 1, 0, 1, 0, 1, 1, 1,   1, 1, 1)


HrsStudying <- data.frame(Hours, Pass)


library(ggplot2)
ggplot(HrsStudying, aes(Hours, Pass)) +
  geom_point(aes()) +
  geom_smooth(method='glm', family="binomial", se=FALSE) +
  labs (x="Hours Studying", y="Probability of Passing Exam",
        title="Probability of Passing Exam vs Hours Studying")

model <- glm(Pass ~.,family=binomial(link='logit'),data=HrsStudying)
model$coefficients
StudentHours <- 2
ProbabilityOfPassingExam <- 1/(1+exp(-(-4.0777+1.5046*StudentHours)))
ProbabilityOfPassingExam
StudentHours <- 4
ProbabilityOfPassingExam <- 1/(1+exp(-(-4.0777+1.5046*StudentHours)))
ProbabilityOfPassingExam

```