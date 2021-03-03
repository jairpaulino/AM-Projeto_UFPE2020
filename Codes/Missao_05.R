
#rm(list=ls())

# Libraries
library(BBmisc)  #normalize
library(caTools) #split
library(e1071)   #svm
library(caret)   #K-fold CV


data = read.table('C:/Users/jairp/Desktop/glass.data', sep=',')
names(data) = c('Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class')
#View(data)

data$Class = as.factor(data$Class) 

# normalize
data[c(2:10)] = normalize(data[c(2:10)], method = "range", 
                 range = c(-1, 1), margin = 2)

data = data[c(2:11)] #View(data)

# split
set.seed(123)
split = sample.split(data$Id, SplitRatio = 0.75)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)
# length(test_set$Class)
# length(training_set$Class)

#ln = length(data$Id)
#per = 0.8
# dataNormTrain = data[1:round((ln*per), 0),]; View(dataNormTrain)
# dataNormTest = data[(round(ln*per, 0)+1):ln,]; View(dataNormTest)


# modelling
modelSVM = svm(formula = Class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# predicting the Test set results
y_pred = predict(modelSVM, newdata = test_set[-10])
confusionMatrix = table(test_set[, 10], y_pred)
cm

# Applying k-Fold Cross Validation
# in creating the folds we specify the target feature (dependent variable) and # of folds
folds = createFolds(training_set$Class, k = 10)
# in cv we are going to applying a created function to our 'folds'
cv = lapply(folds, function(x){ # start of function
  # in the next two lines we will separate the Training set into it's 10 pieces
  training_fold = training_set[-x, ] # training fold =  training set minus (-) it's sub test fold
  test_fold = training_set[x, ] # here we describe the test fold individually
  # now apply (train) the classifer on the training_fold
  classifier = svm(formula = Class ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  # next step in the loop, we calculate the predictions and cm and we equate the accuracy
  # note we are training on training_fold and testing its accuracy on the test_fold
  y_pred = predict(modelSVM, newdata = test_fold[-10])
  cm = table(test_fold[, 10], y_pred)
  return(cm)
})
  
# knitr::include_graphics("CV.png")

cv

eval <- evaluate(cv[1],
                 target_col = "target",
                 prediction_cols = "prediction",
                 type = "binomial")


# https://rpubs.com/markloessi/506713