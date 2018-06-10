"
if ('caret' %in% rownames(installed.packages()) == FALSE) {
    install.packages('caret', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(caret)
if ('tree' %in% rownames(installed.packages()) == FALSE) {
    install.packages('tree', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(tree)
"

if ('rpart' %in% rownames(installed.packages()) == FALSE) {
    install.packages('rpart', repos='http://healthstat.snu.ac.kr/CRAN/')
}
library(rpart)

df <- read.csv('./datasets/german_credit/german_credit.csv')    # header=FALSE
str(df)

y <- as.factor(df[,1])
x <- df[,2:ncol(df)]
fit <- rpart(y~., x)    # .: All variables, refer Formula Notations
1 - sum(y == predict(fit, x, type="class")) / length(y) # = training error

fit
plot(fit)
text(fit)