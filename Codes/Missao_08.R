# Algorithm: Classificacao da base de dados Glass
# Authors: 
# Date: 

#rm(list=ls())

# Libraries
library(BBmisc)  #normalize
library(caTools) #split
library(e1071)   #svm
library(caret)   #K-fold CV
library(rpart)
library(tree)

data = read.table('C:/Users/jairp/Desktop/glass.data', sep=',')
names(data) = c('Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class')
#View(data)

data$Class = as.factor(data$Class) 

# normalize
data[c(2:10)] = normalize(data[c(2:10)], 
                          method = "range", 
                 range = c(0.2, 0.8), margin = 2)

data = data[c(2:11)] #View(data)

# split
set.seed(123)
split = sample.split(data$RI, SplitRatio = 0.8)
training_set = subset(data, split == TRUE) # length(test_set$Class) - 43
test_set = subset(data, split == FALSE) # length(training_set$Class) - 171

# Modelling
# KNN
trControl.knn <- trainControl(method  = "repeatedcv",
                          number  = 5)

fit.knn <- train(Class ~ .,
           method     = "knn",
           tuneGrid   = expand.grid(k = 1:50),
           trControl  = trControl.knn,
           metric     = "Accuracy",
           data       = training_set)

result.knn = predict(fit.knn, newdata = training_set[c(-10)])
table(result.knn, training_set$Class)
confusionMatrix(result.knn, training_set$Class)

# svm
fit.svm = svm(formula = Class  ~ .,
              data = training_set,
              type = 'C-classification',
              kernel = 'radial')

result.svm = predict(fit.svm, newdata = training_set[c(-10)])
table(as.numeric(result.svm), training_set$Class)
confusionMatrix(as.factor(result.svm), training_set$Class)


# ANN
training_set$Class = as.factor(training_set$Class)
control <- trainControl(method="repeatedcv", number=5, repeats=1)
fit.ann <- train(Class ~ ., 
               data = training_set,
               method="neuralnet", 
               algorithm = "backprop", 
               learningrate = 0.25, 
               threshold = 0.1, 
               trControl = control)

result.ann = predict(fit.ann, newdata = training_set[c(-10)])
table(as.numeric(result.ann), training_set$Class)



fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)



model_rf <- train(training_set[c(-10)],
                  training_set[c(10)],
                  method='rf',
                  trControl=fitControl,
                  tuneLength=3)

