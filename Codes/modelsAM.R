# M1 - Classificador bayesiano gaussiano (CBG)
getGBG = function(train, test, exportResults = F){
  #train = dataNormTrain; test = dataNormTest
  
  model_cbg = naiveBayes(x = train[-5], 
                   y = train$class)
  #print(model_cbg)
  
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

# M2 - Classificador bayesiano gaussiano (CBG)
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
                   alpha = 0, #lasso
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
