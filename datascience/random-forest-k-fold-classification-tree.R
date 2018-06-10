"
if ('caret' %in% rownames(installed.packages()) == FALSE) {
    install.packages('caret', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(caret)
if ('tree' %in% rownames(installed.packages()) == FALSE) {
    install.packages('tree', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(tree)
if ('randomForest' %in% rownames(installed.packages()) == FALSE) {
    install.packages('randomForest', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(randomForest)
"

if ('rpart' %in% rownames(installed.packages()) == FALSE) {
    install.packages('rpart', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(rpart)

df <- read.csv('./datasets/german_credit/german_credit.csv')
str(df)

#treemod <- tree()
#glimpse(df)

# cross validation, using random forest to predict
k = 5

df$id <- sample(1:k, nrow(df), replace=TRUE)
list <- 1:k

# prediction and test set data frames that we add to with each iteration over the folds
prediction <- data.frame()
testsetCopy <- data.frame()

#progress.bar <- create_progress_bar("Progress")
#progress.bar$init(k)

for (i in 1:k) {
    # remove rows with id i from dataframe to create training set
    # select rows with id i to create test set
    training.set <- subset(df, id %in% list[-i])
    test.set <- subset(df, id %in% c(i))

    # run random forest model
    model <- randomForest(training.set$Creditability ~ ., data=training.set, ntree=100)

    # remove response column
    temp <- as.data.frame(predict(model, test.set[,-1]))

    (i)
    summary(temp)

    # append this iteration's predictions to the end of the prediction data frame
    prediction <- rbind(prediction, temp)

    # append this iteration's test set to the test set copy data frame
    testsetCopy <- rbind(testsetCopy, as.data.frame(test.set[,1]))
}

# add predictions and actual values
result <- cbind(prediction, testsetCopy[,1])
names(result) <- c("Predicted", "Actual")
result$Difference <- abs(result$Actual - result$Predicted)

# Mean Absolute Error
summary(result$Difference)