library(caret)
library(knitr)
library(randomForest)


training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

trainingNzv <- training[,!nearZeroVar(training, saveMetrics = TRUE)$nzv]
testingNzv <- testing[,!nearZeroVar(training, saveMetrics = TRUE)$nzv]

colNa <- colSums(is.na(trainingNzv)) < 1

trainingNzvNa <- trainingNzv[,colNa]
testingNzvNa <- testingNzv[,colNa]

trainingNzvNa <- trainingNzvNa[,-c(1:6)]
testingNzvNa <- testingNzvNa[,-c(1:6)]

set.seed(123)

inTrain <- createDataPartition(y=trainingNzvNa$classe, p=0.7, list=FALSE)

trainingPart <- trainingNzvNa[inTrain,]
testingPart <- trainingNzvNa[-inTrain,]

set.seed(12345)
modelFit <- train(classe ~ ., data=trainingPart, method = "gbm", verbose=FALSE)

set.seed(12345)
modelFitRf <- train(classe ~ ., data=trainingPart, method = "rf", 
                    trControl = trainControl(method="repeatedcv"), 
                    number = 10, repeats = 10)

set.seed(12345)
modelFitLda <- train(classe ~ ., data=trainingPart, method = "lda")

trainPred <- predict(modelFit, newdata = trainingPart)
trainPredRf <- predict(modelFitRf, newdata = trainingPart)
trainPredLda <- predict(modelFitLda, newdata = trainingPart)

comboDf <- data.frame(trainPred, 
                      trainPredRf,
                      trainPredLda,
                      classe = trainingPart$classe)

set.seed(12345)
modelFitCombo <- train(classe ~ ., data=comboDf, method = "rf", 
                       trControl = trainControl(method="repeatedcv"), 
                       number = 10, repeats = 10)

testPred <- predict(modelFit, newdata = testingPart)
testPredRf <- predict(modelFitRf, newdata = testingPart)
testPredLda <- predict(modelFitLda, newdata = testingPart)

testComboDf <- data.frame(trainPred = testPred, 
                          trainPredRf = testPredRf, 
                          trainPredLda = testPredLda,
                          classe = testingPart$classe)

testPredCombo <- predict(modelFitCombo, newdata = testComboDf)

cat("GBM")
confusionMatrix(testPred, testingPart$classe)

cat("RF")
confusionMatrix(testPredRf, testingPart$classe)

cat("Lda")
confusionMatrix(testPredLda, testingPart$classe)

cat("Combo")
confusionMatrix(testPredCombo, testingPart$classe)

testResult <- predict(modelFit, newdata = testingNzvNa)
testResultRf <- predict(modelFitRf, newdata = testingNzvNa)
testResultLda <- predict(modelFitLda, newdata = testingNzvNa)

testNzvNaComboDf <- data.frame(trainPred = testResult, 
                               trainPredRf = testResultRf, 
                               trainPredLda = testResultLda, 
                               classe = rep(0,20))

testResulCombo <- predict(modelFitCombo, newdata = testNzvNaComboDf)

testResult
testResultRf
testResultLda
testResulCombo

