if ('rpart' %in% rownames(installed.packages()) == FALSE) {
    install.packages('rpart', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(rpart)

train <- read.csv('./datasets/sonar/train.csv', header=FALSE)
#nrow(train)
y <- as.factor(train[, 61])
x <- train[, 1:60]
fit <- rpart(y~., x)    # .: All variables, refer R Formula Notations (+ control=rpart.control(maxdepth=1))
training_error <- 1 - sum(y == predict(fit, x, type='class')) / length(y) # training error

fit
plot(fit)
text(fit)
training_error

test <- read.csv('./datasets/sonar/test.csv', header=FALSE)
y_test <- as.factor(test[, 61])
x_test <- test[, 1:60]
testing_error <- 1 - sum(y_test == predict(fit, x_test, type='class')) / length(y_test)
testing_error

