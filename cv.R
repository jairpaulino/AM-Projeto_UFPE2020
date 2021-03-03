
foldIndex = cvFolds(length(dataNormTrain$vwti), K = 10, R = 1)
resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
for(i in 1:k){#i=1
  train <- dataNormTrain[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
  validation <- dataNormTrain[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
  
  model_cbg_cv = naiveBayes(x = dataNormTrain[-5], 
                         y = dataNormTrain$class)
  
  cgb_predict = predict(model_cbg_cv, newdata = validation[-5])
  cbgMetrics = getMetrics(cgb_predict, validation$class)
  resultMatrixCV[i,] = cbgMetrics
}

write.csv(resultMatrixCV, file = "Results/cv_metrics_cbg.csv")
