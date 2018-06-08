require(randomForest)

data(iris)
set.seed(1)

dat <- iris

dat$Species <- factor(ifelse(dat$Species == 'virginica', 'virginica', 'other'))

model.rf <- randomForest(Species~., dat, ntree=25, importance=True, nodesize=5)

model.rf

varImpPlot(model.rf)