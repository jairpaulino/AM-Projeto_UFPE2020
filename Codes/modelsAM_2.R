# M1 - Classificador bayesiano gaussiano (CBG)
getCBG = function(train, test, exportResults = F){
  #train = dataNormTrain; test = dataNormTest
  
  model_cbg = naiveBayes(x = train[-5], y = train$class)
  #plot(model_cbg)
  
  cgb_predict = predict(model_cbg, newdata = test[-5])
  matriz_CBG = table(test$class, cgb_predict)
  
  confusionMatrix(matriz_CBG, positive = "1")
  
  cbgMetrics = getMetrics(cgb_predict, test$class)
  
  if(exportResults == T){
    write.csv(cbgMetrics, file = "Results/metrics_cbg.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test[[1]]), ncol=2))
  names(result_df) = c('Target', 'CBG')
  result_df$Target = test$class
  result_df$CBG = cgb_predict
  
  return(list('Model'= model_cbg,
              'Metrics'= cbgMetrics,
              'Results'= result_df
              )
  )
}

# M1_cv - Classificador bayesiano gaussiano (CBG) com CV
getCBG_cv = function(train_df, test_df, exportResults = F){
  #train_df = dataNormTrain; test_df = dataNormTest
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=9
    train <- train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation <- train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    
    model_cbg_cv = naiveBayes(x = train[-5], 
                              y = train$class)
    
    cgb_predict = predict(model_cbg_cv, newdata = validation[-5])
    cbgMetrics = getMetrics(cgb_predict, validation$class)
    resultMatrixCV[i,] = cbgMetrics
    
    #View(resultMatrixCV)
    if(cbgMetrics[1] < bestModelAcc){
      bestModelAcc = cbgMetrics[1]
      bestModelIndex = i
      bestModel = model_cbg_cv
    }
  }
  
  write.csv(resultMatrixCV, file = "Results/cv_metrics_cbg.csv")

  # Calculo do IC
  
  meanErroRateresult = mean(resultMatrixCV$ErroRate)
  sdErroRateresult = sd(resultMatrixCV$ErroRate)
  icErroRate = (t.test(resultMatrixCV$ErroRate))$conf.int[1:2]
  
  meanPrecision = mean(resultMatrixCV$Precision)
  sdPrecision = sd(resultMatrixCV$Precision)
  icPrecision = (t.test(resultMatrixCV$Precision))$conf.int[1:2]
  
  meanRecall = mean(resultMatrixCV$recall)
  sdRecall = sd(resultMatrixCV$recall)
  icRecall = (t.test(resultMatrixCV$recall))$conf.int[1:2]
  
  meanf1Score = mean(resultMatrixCV$F1)
  sdf1Score = sd(resultMatrixCV$F1)
  icf1Score = (t.test(resultMatrixCV$F1))$conf.int[1:2]
  
  icCBG = as.data.frame(matrix(nrow = 4, ncol=2))
  names(icCBG) = c('Inf', 'Sup')
  icCBG[1,] = icErroRate
  icCBG[2,] = icPrecision
  icCBG[3,] = icRecall
  icCBG[4,] = icf1Score
  
  write.csv(icCBG, file = "Results/cbg_ic.csv")
  
  # Treinamento
  #model_cbg_train = naiveBayes(x = train_df[-5], y = train_df$class)
  cgb_predict = predict(bestModel, newdata = train_df[-5])
  matriz_CBG = table(train_df$class, cgb_predict)
  cbgMetrics_train = getMetrics(cgb_predict, train_df$class)
  
  # Teste
  cgb_predict = predict(bestModel, newdata = test_df[-5])
  matriz_CBG = table(test_df$class, cgb_predict)
  cbgMetrics = getMetrics(cgb_predict, test_df$class)
  
  # IC para teste
  resultIC = as.data.frame(matrix(ncol=4, nrow=3))
  names(resultIC) = c('ErroRate', 'Precision', 'recall', 'F1')
  ##rownames(resultIC) = c('Métrica', 'Sup.', 'Inf.')
  resultIC$ErroRate[1] = cbgMetrics[1]
  resultIC$ErroRate[2] = cbgMetrics$ErroRate - (2.262*sdErroRateresult)/sqrt(10)
  resultIC$ErroRate[3] = cbgMetrics$ErroRate + (2.262*sdErroRateresult)/sqrt(10)
  resultIC$Precision[1] = cbgMetrics[2]
  resultIC$Precision[2] = cbgMetrics$Precision - (2.262*sdPrecision)/sqrt(10)
  resultIC$Precision[3] = cbgMetrics$Precision + (2.262*sdPrecision)/sqrt(10)
  resultIC$recall[1] = cbgMetrics[3]
  resultIC$recall[2] = cbgMetrics$recall - (2.262*sdRecall)/sqrt(10)
  resultIC$recall[3] = cbgMetrics$recall + (2.262*sdRecall)/sqrt(10)
  resultIC$F1[1] = cbgMetrics[4]
  resultIC$F1[2] = cbgMetrics$F1 - (2.262*sdRecall)/sqrt(10)
  resultIC$F1[3] = cbgMetrics$recall + (2.262*sdRecall)/sqrt(10)
  
  if(exportResults == T){
    write.csv(cbgMetrics, file = "Results/metrics_cbg.csv")
    #write.csv(resultIC, file = "Results/cbg_resultIC.csv")
  }
  result_df = as.data.frame(matrix(nrow = length(test_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'CBG')
  result_df$Target = test_df$class
  result_df$CBG = cgb_predict
  
  return(list('Model'= bestModel,
              'Metrics'= cbgMetrics,
              'Results'= result_df
  )
  )
}

# M2 - Classificador bayesiano - KNN
getKNN = function(train, valid, test, exportResults = F){
  #train = dataNormTrain; valid = dataNormValid; test = dataNormTest
  
  # Setting the Cross-validation 10-fold to kNN and LVQ
  trControl = trainControl(method = "cv",
                           number = 10
  )
  
  set.seed(123)
  model_knn = train(class ~ ., 
                    data = valid,
                    method = 'knn', 
                    trControl = trControl,
                    tuneGrid   = expand.grid(k = 1:50),
                    #preProcess = c("center", "scale"),
                    metric = "Accuracy",
                    #tuneLength = 50
                    
  )
  
  #model_knn
  
  knn_predict <- predict(model_knn, newdata = test[-5])
  matriz_knn = table(test$class, knn_predict)
  
  confusionMatrix(knn_predict, test$class)
  knnMetrics = getMetrics(knn_predict, test$class)
  
  if(exportResults == T){
    write.csv(knnMetrics, file = "Results/metrics_knn.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(dataNormTest[[1]]), ncol=2))
  names(result_df) = c('Target', 'KNN')
  result_df$Target = test$class
  result_df$KNN = knn_predict
  
  return(list('Model'= model_knn,
              'Metrics'= knnMetrics,
              'Results'= result_df)
  )
}

# M2_cv - Classificador bayesiano - KNN com CV
getKNN_cv = function(train_df, valid_df, test_df, exportResults = F){
  #train_df = dataNormTrain; valid_df = dataNormValid; test_df = dataNormTest
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    previsoes = NULL; metricFI = NULL
    maxF1 = 0; maxI = NULL
    for(j in 1:20){#j=50
      set.seed(1)
      previsoes = knn(train = train[-5], 
                      test = validation[-5],
                      cl = train$class,
                      k = j)
      
      metricFI[j] = getMetrics(previsoes, validation$class)$F1 
      
      if(metricFI[j] > maxF1){
        maxF1 = metricFI[j] 
        maxI = j
      }

    knn_Predict = knn(train = train[-5], 
                    test = validation[-5],
                    cl = train$class,
                    k = maxI)
    
    knnMetrics = getMetrics(knn_Predict, validation$class)
    resultMatrixCV[i,] = knnMetrics
    
    #View(resultMatrixCV)
    if(knnMetrics[1] < bestModelAcc){
      bestModelAcc = knnMetrics[1]
      bestModelIndex = i
      }
    }
}

  knnPredictTest = knn(train = train_df[-5], 
                       test = test_df[-5],
                       cl = train_df$class,
                       k = maxI)
  
  knnMetricsTest = getMetrics(knnPredictTest, test_df$class)


  if(exportResults == T){
    write.csv(knnMetricsTest, file = "Results/knn_metrics.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'KNN')
  result_df$Target = test_df$class
  result_df$KNN = knnPredictTest
  
  return(list('K'= maxI,
              'Metrics'= knnMetrics,
              'Results'= result_df)
  )
}

# M3 - CBG baseado em Janela de Parzen (CBG-JP) 

# M4 - Regressão Logística (RL)
getRL = function(train, test, exportResults = F){
  #train = dataNormTrain; test = dataNormTest
  
  model_lr = glm(class ~ ., 
                 data = dataNormTrain,
                 family = binomial(link = 'logit')
  )
  
  #model_lr
  
  lr_predict = predict(model_lr, 
                       newdata = test[-5],
                       type = "response")
  
  lr_predict = ifelse(lr_predict < 0.5, 0, 1)
  
  matriz_lr = table(test$class, lr_predict)
  
  confusionMatrix(matriz_lr, positive = "1")
  
  lrMetrics = getMetrics(lr_predict, test$class)
  
  if(exportResults == T){
    write.csv(lrMetrics, file = "Results/metrics_lr.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test[[1]]), ncol=2))
  names(result_df) = c('Target', 'LR')
  result_df$Target = test$class
  result_df$LR = lr_predict
  
  return(list('Model'= model_lr,
              'Metrics'= lrMetrics,
              'Results'= result_df)
  )
}

# M5 - Regressão logistica com Regularizacao (RLR) 
getLRR= function(train, valid, test, exportResults = F){
  #train = dataNormTrain; valid = dataNormValid; test = dataNormTest
  
  train$x5 = (train$vwti)^2
  train$x6 = sqrt(train$swti) 
  train$x7 = log(train$cwti) 
  train$x8 = (train$ei)^(-1) 
  
  valid$x5 = (valid$vwti)^2
  valid$x6 = sqrt(valid$swti) 
  valid$x7 = log(valid$cwti) 
  valid$x8 = (valid$ei)^(-1) 
  
  test$x5 = (test$vwti)^2
  test$x6 = sqrt(test$swti) 
  test$x7 = log(test$cwti) 
  test$x8 = (test$ei)^(-1) 
  
  train_val_y = as.data.frame(valid[-5]) 
  train_val_x = (valid$class)
  
  # RLR - cross-validation 
  set.seed(123)
  cvLR = cv.glmnet(x = as.matrix(train_val_y), #lr_reg_ind
                   y = train_val_x, #lr_reg_dep
                   family = "binomial",
                   type.measure = "deviance", 
                   nfolds = 10,
                   alpha = 0.5, #lasso
                   parallel = F)
  
  
  # Treinamento
  lr_reg_train_y = train[-5]
  lr_reg_train_x = train$class
  
  model_lrr = glmnet(x = as.matrix(lr_reg_train_y), 
                  y = lr_reg_train_x,
                  family = "binomial",
                  lambda = cvLR$lambda.1se)
  
  lr_reg_test_y = test[-5]
  lr_reg_test_x = test$class
  
  lrr_predict = predict(model_lrr,
                     s = cvLR$lambda.1se,
                     newx = as.matrix(lr_reg_test_y),
                     type = "response")
  
  lrr_predict = as.factor(ifelse(lrr_predict < 0.5, 0, 1))
  
  matriz_knn = table(test$class, lrr_predict)
  
  confusionMatrix(lrr_predict, test$class)

  lrrMetrics = getMetrics(lrr_predict, test$class)
  
  if(exportResults == T){
    write.csv(lrrMetrics, file = "Results/metrics_lrr.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test[[1]]), ncol=2))
  names(result_df) = c('Target', 'LRR')
  result_df$Target = test$class
  result_df$LRR = lrr_predict
  
  return(list('Model'= model_lrr,
              'Metrics'= lrrMetrics,
              'Results'= result_df)
  )
}

# M6 - Ensemble com regra do voto majoritario (EVM)
getEVM = function(classResult, exportResults = F){
  
  classResult_evm = as.matrix(classResult)
  
  for (i in 1:length(dataNormTest$class)){ #i=1
    nVote = sum(as.numeric(classResult_evm[i, 2:5]))
    
    if (nVote > 2){
      classResult$EVM[i] = 1
    } else {
      classResult$EVM[i] = 0
    }
    
  } #View(classResult)
  
  matriz_evm = table(classResult$Target, classResult$EVM)
  
  evm_regMetrics = getMetrics(classResult$EVM, classResult$Target)

  if(exportResults == T){
    write.csv(evm_regMetrics, file = "Results/metrics_evm.csv")
  }
  
  return(list('Metrics'= evm_regMetrics,
              'Results'= classResult)
  )
}
