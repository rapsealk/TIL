#!/usr/bin/env Rscript
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

train <- read.csv('./titanic-data/train.csv', sep=',', header=T, na.strings=c(''))
# test <- read.csv('./titanic-data/test.csv', sep=',', header=T, na.strings=c(''))
#str(train)

train$Survived <- as.factor(train$Survived)
survived <- nrow(train[train$Survived == 1,])
died <- nrow(train[train$Survived == 0,])
survival_rate <- round(survived / (survived + died) * 100.0, digits=2)
survival_rate
(survival_frame <- data.frame(x=factor(c('Survived', 'Died')), y=c(survived, died)))
#hist(survival_frame)
train2 = data.frame(Survived=train$Survived, Pclass=train$Pclass, Sex=train$Sex, Age=train$Age, Fare=train$Fare)
sapply(data.frame(Age=train$Age, Fare=train$Fare), mean, na.rm=TRUE)
#train_sex = data.frame(x=factor(c('male', 'female')), y=c(nrow(train[train$Sex == male]), nrow(train[train$Sex == female])))
#sapply(train2, mean, na.rm=TRUE)
fivenum(train$Fare) # 1) sample minimum, 2) first quartile, 3) median, 4) third quartile, 5) sample maximum
#library(Hmisc)
#describe(train2)
hist(train$Fare, freq=FALSE)
#sapply(train$Age, mean, na.rm=TRUE)
#train$Pclass <- as.factor(train$Pclass)
#table(train$Pclass)
# test$Pclass <- as.factor(test$Pclass)
# table(train$Pclass)

