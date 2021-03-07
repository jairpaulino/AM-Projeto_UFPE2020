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
  
  write.csv(resultMatrixCV, file = "Results/cbg_cv_metrics.csv")

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
    write.csv(cbgMetrics, file = "Results/cbg_metrics.csv")
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

# M2_cv - Classificador bayesiano - KNN com CV
getKNN_cv = function(train_df, test_df, exportResults = F){
  #train_df = dataNormTrain; valid_df = dataNormValid; test_df = dataNormTest
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    k = 1
    previsoes = NULL; metricFI = NULL
    maxF1 = 0; maxI = NULL
    
    for(j in k:30){#j=50
      set.seed(1)
      previsoes = knn(train = train[-5], 
                      test = validation[-5],
                      cl = train$class,
                      k = j)
      
      
      resultMatrixCV[i,] = getMetrics(previsoes, validation$class)
      write.csv(resultMatrixCV, file = "Results/knn_cv_metrics.csv")
      
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
  
  # Calculo do IC
  meanErroRateresult = mean(resultMatrixCV$ErroRate)
  sdErroRateresult = sd(resultMatrixCV$ErroRate)
  icErroRate = (t.test(resultMatrixCV$ErroRate))$conf.int[1:2]
  
  meanPrecision = mean(resultMatrixCV$Precision)
  sdPrecision = sd(resultMatrixCV$Precision)
  icPrecision = (t.test(resultMatrixCV$Precision))$conf.int[1:2]
  
  meanRecall = mean(resultMatrixCV$recall)
  sdRecall = sd(resultMatrixCV$recall)
  #icRecall = (t.test(resultMatrixCV$recall))$conf.int[1:2]
  
  meanf1Score = mean(resultMatrixCV$F1)
  sdf1Score = sd(resultMatrixCV$F1)
  icf1Score = (t.test(resultMatrixCV$F1))$conf.int[1:2]
  
  icKnn = as.data.frame(matrix(nrow = 4, ncol=2))
  names(icKnn) = c('Inf', 'Sup')
  icKnn[1,] = icErroRate
  icKnn[2,] = icPrecision
  icKnn[3,] = 1#icRecall
  icKnn[4,] = icf1Score
  
  write.csv(icKnn, file = "Results/knn_ic.csv")
  
  # Treinamento
  knnPredictTest = knn(train = train_df[-5], 
                       test = test_df[-5],
                       cl = train_df$class,
                       k = maxI)
 
  knnMetricsTest = getMetrics(knnPredictTest, test_df$class)

  if(exportResults == T){
    write.csv(knnMetricsTest, file = "Results/knn_metrics.csv")
  }
  
  # IC para teste
  resultIC = as.data.frame(matrix(ncol=4, nrow=3))
  names(resultIC) = c('ErroRate', 'Precision', 'recall', 'F1')
  ##rownames(resultIC) = c('Métrica', 'Sup.', 'Inf.')
  resultIC$ErroRate[1] = knnMetricsTest[1]
  resultIC$ErroRate[2] = knnMetrics$ErroRate - (2.262*sdErroRateresult)/sqrt(10)
  resultIC$ErroRate[3] = knnMetrics$ErroRate + (2.262*sdErroRateresult)/sqrt(10)
  resultIC$Precision[1] = knnMetricsTest[2]
  resultIC$Precision[2] = knnMetrics$Precision - (2.262*sdPrecision)/sqrt(10)
  resultIC$Precision[3] = knnMetrics$Precision + (2.262*sdPrecision)/sqrt(10)
  resultIC$recall[1] = knnMetricsTest[3]
  resultIC$recall[2] = knnMetrics$recall - (2.262*sdRecall)/sqrt(10)
  resultIC$recall[3] = knnMetrics$recall + (2.262*sdRecall)/sqrt(10)
  resultIC$F1[1] = knnMetricsTest[4]
  resultIC$F1[2] = knnMetrics$F1 - (2.262*sdRecall)/sqrt(10)
  resultIC$F1[3] = knnMetrics$recall + (2.262*sdRecall)/sqrt(10)
  
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
getParzen_cv = function(train_df, test_df, exportResults = T ) {
  #train_df = dataNormTrain; test_df = dataNormTest
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=5, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1', 'h')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=3
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    
    count_0 = sum(train[,5] == 1)
    count_1 = sum(train[,5] == 0)
    
    count_total = count_0+count_1
    f_0 = count_0/count_total
    f_1 = count_1/count_total
    
    sample = train[,5] == 0
    data_0 = subset(train, sample == T)
    data_1 = subset(train, sample == F)
    
    zero_X = data_0[, 1:4]
    zero_y = data_0[, 5]
    
    hum_X = data_1[, 1:4]
    hum_y = data_1[, 5]
    
    h = c(0.01, 0.1, 0.5, 1, 1.25); 
    bestH = 0; bestMetric = 0
    for (j in 1:5){#j=1
      
      H <- diag(c(h[j], h[j], h[j], h[j]))
      zeroModel = kde(x=zero_X[,1:4], H = H)
      
      humModel = kde(x=hum_X[,1:4], H = H)
      
      ParzenClf_valid = NULL
      for(k in 1:length(validation$class)){#k=1
        
        prob_0 = predict(zeroModel, x = validation[k, 1:4])
        prob_ap_0 = prob_0*f_0
        
        prob_1 = predict(humModel, x = validation[k, 1:4])
        prob_ap_1 = prob_1*f_1
        
        if(prob_0 > prob_1){
          ParzenClf_valid[k] = 0
        }else{
          ParzenClf_valid[k] = 1
        }  
      }
      
      metricValid = getMetrics(ParzenClf_valid, validation$class)
      
      print(paste('Iter.:', i, j, h[j], metricValid[4], sep = ' '))
      
      if(metricValid[4] > bestMetric){
        bestMetric = metricValid[4] 
        bestH = j
      }
    }
    
    resultMatrixCV[i,1:4] = metricValid
    resultMatrixCV[i, 5] = h[bestH] 
  }
  
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
  
  icParzen = as.data.frame(matrix(nrow = 4, ncol=2))
  names(icParzen) = c('Inf', 'Sup')
  icParzen[1,] = icErroRate
  icParzen[2,] = icPrecision
  icParzen[3,] = icRecall
  icParzen[4,] = icf1Score
  
  write.csv(icParzen, file = "Results/parzen_ic.csv")
  
  # Treinamento
  count_0_train = sum(train_df[,5] == 1)
  count_1_train = sum(train_df[,5] == 0)
  
  count_total_train = count_0_train+count_1_train
  f_0 = count_0_train/count_total_train
  f_1 = count_1_train/count_total_train
  
  sample = train_df[,5] == 0
  data_0_train = subset(train_df, sample == T)
  data_1_train = subset(train_df, sample == F)
  
  zero_X_train = data_0_train[, 1:4]
  zero_y_train = data_0_train[, 5]
  
  hum_X_train = data_1_train[, 1:4]
  hum_y_train = data_1_train[, 5]
  
  bestFold = subset(resultMatrixCV, F1 == max(resultMatrixCV$F1))
  bestH = bestFold$h
  H = diag(c(bestH, bestH, bestH, bestH))
  zeroModel = kde(x=zero_X_train[,1:4], H = H)
  humModel = kde(x=hum_X_train[,1:4], H = H)
  
  parzenClf_train = NULL
  for(i in 1:length(train_df$class)){#i=1
    prob_0 = predict(zeroModel, x = train_df[i, 1:4])
    prob_ap_0 = prob_0*f_0
    prob_1 = predict(humModel, x = train_df[i, 1:4])
    prob_ap_1 = prob_1*f_1
    
    if(prob_0 > prob_1){
      parzenClf_train[i] = 0
    }else{
      parzenClf_train[i] = 1
    }  
  }
  
  #matriz_parzen = table(train_df$class, parzenClf_train)
  parzenMetrics_train = getMetrics(parzenClf_train, train_df$class)
  
  # Teste
  parzenClf_test = NULL
  for(i in 1:length(test_df$class)){#i=1
    prob_0 = predict(zeroModel, x = test_df[i, 1:4])
    prob_ap_0 = prob_0*f_0
    prob_1 = predict(humModel, x = test_df[i, 1:4])
    prob_ap_1 = prob_1*f_1
    
    if(prob_0 > prob_1){
      parzenClf_test[i] = 0
    }else{
      parzenClf_test[i] = 1
    }  
  }
  
  #matriz_parzen = table(test_df$class, parzenClf_test)
  parzenMetrics_test = getMetrics(parzenClf_test, test_df$class)
  
  # IC para teste
  resultIC = as.data.frame(matrix(ncol=4, nrow=3))
  names(resultIC) = c('ErroRate', 'Precision', 'recall', 'F1')
  ##rownames(resultIC) = c('Métrica', 'Sup.', 'Inf.')
  resultIC$ErroRate[1] = parzenMetrics_test[1]
  resultIC$ErroRate[2] = parzenMetrics_test$ErroRate - (2.262*sdErroRateresult)/sqrt(10)
  resultIC$ErroRate[3] = parzenMetrics_test$ErroRate + (2.262*sdErroRateresult)/sqrt(10)
  resultIC$Precision[1] = parzenMetrics_test[2]
  resultIC$Precision[2] = parzenMetrics_test$Precision - (2.262*sdPrecision)/sqrt(10)
  resultIC$Precision[3] = parzenMetrics_test$Precision + (2.262*sdPrecision)/sqrt(10)
  resultIC$recall[1] = parzenMetrics_test[3]
  resultIC$recall[2] = parzenMetrics_test$recall - (2.262*sdRecall)/sqrt(10)
  resultIC$recall[3] = parzenMetrics_test$recall + (2.262*sdRecall)/sqrt(10)
  resultIC$F1[1] = parzenMetrics_test[4]
  resultIC$F1[2] = parzenMetrics_test$F1 - (2.262*sdRecall)/sqrt(10)
  resultIC$F1[3] = parzenMetrics_test$recall + (2.262*sdRecall)/sqrt(10)
  
  if(exportResults == T){
    write.csv(parzenMetrics_test, file = "Results/parzem_metrics.csv")
    #write.csv(resultIC, file = "Results/cbg_resultIC.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'Parzen')
  result_df$Target = test_df$class
  result_df$Parzen = parzenClf_test
  
  return(list('h'= bestH,
              'Metrics'= parzenMetrics_test,
              'Results'= result_df
  )
  )
  
}

# M4 - Regressão Logística (RL)
getRL_cv = function(train_df, test_df, exportResults = F){
  #train_df = dataNormTrain; test_df = dataNormTest
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=1
    train <- train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation <- train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    
    model_lr_cv = glm(class ~ .,
                      data = train,
                      family = binomial(link = 'logit')
    )
    
    lr_predict = predict(model_lr_cv, newdata = validation[-5])
    
    lr_predict = ifelse(lr_predict < 0.5, 0, 1)
    
    lrMetrics = getMetrics(as.numeric(lr_predict), validation$class)
    resultMatrixCV[i,] = lrMetrics
    
    #View(resultMatrixCV)
    if(lrMetrics[1] < bestModelAcc){
      bestModelAcc = lrMetrics[1]
      bestModelIndex = i
      bestModel = model_lr_cv
    }
  }
  
  write.csv(resultMatrixCV, file = "Results/rl_cv_metrics.csv")
  
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
  
  icLR = as.data.frame(matrix(nrow = 4, ncol=2))
  names(icLR) = c('Inf', 'Sup')
  icLR[1,] = icErroRate
  icLR[2,] = icPrecision
  icLR[3,] = icRecall
  icLR[4,] = icf1Score
  
  write.csv(icLR, file = "Results/lr_ic.csv")
  
  # Treinamento
  lr_predict_train = predict(bestModel, newdata = train_df[-5])
  lr_predict_train = ifelse(lr_predict_train < 0.5, 0, 1)
  matriz_lr = table(train_df$class, lr_predict_train)
  lrMetrics_train = getMetrics(lr_predict_train, train_df$class)
  
  # Teste
  lr_predict_test = predict(bestModel, newdata = test_df[-5])
  lr_predict_test = ifelse(lr_predict_test < 0.5, 0, 1)
  lrMetrics = getMetrics(lr_predict_test, test_df$class)
  
  # IC para teste
  resultIC = as.data.frame(matrix(ncol=4, nrow=3))
  names(resultIC) = c('ErroRate', 'Precision', 'recall', 'F1')
  ##rownames(resultIC) = c('Métrica', 'Sup.', 'Inf.')
  resultIC$ErroRate[1] = lrMetrics[1]
  resultIC$ErroRate[2] = lrMetrics$ErroRate - (2.262*sdErroRateresult)/sqrt(10)
  resultIC$ErroRate[3] = lrMetrics$ErroRate + (2.262*sdErroRateresult)/sqrt(10)
  resultIC$Precision[1] = lrMetrics[2]
  resultIC$Precision[2] = lrMetrics$Precision - (2.262*sdPrecision)/sqrt(10)
  resultIC$Precision[3] = lrMetrics$Precision + (2.262*sdPrecision)/sqrt(10)
  resultIC$recall[1] = lrMetrics[3]
  resultIC$recall[2] = lrMetrics$recall - (2.262*sdRecall)/sqrt(10)
  resultIC$recall[3] = lrMetrics$recall + (2.262*sdRecall)/sqrt(10)
  resultIC$F1[1] = lrMetrics[4]
  resultIC$F1[2] = lrMetrics$F1 - (2.262*sdRecall)/sqrt(10)
  resultIC$F1[3] = lrMetrics$F1 + (2.262*sdRecall)/sqrt(10)
  
  if(exportResults == T){
    write.csv(lrMetrics, file = "Results/rl_metrics.csv")
    #write.csv(resultIC, file = "Results/cbg_resultIC.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'LR')
  result_df$Target = test_df$class
  result_df$LR = lr_predict_test
  
  return(list('Model'= bestModel,
              'Metrics'= lrMetrics,
              'Results'= result_df)
  )
}

# M5 - Regressão logistica com Regularizacao (RLR) 
getRLR_cv= function(train_df, test_df, exportResults = F){
  #train_df = dataNormTrain; test_df = dataNormTest

  train_df$x5 = (train_df$vwti)^2
  train_df$x6 = sqrt(train_df$swti) 
  train_df$x7 = log(train_df$cwti) 
  train_df$x8 = (train_df$ei)^(-1) 
  
  # valid_df$x5 = (valid_df$vwti)^2
  # valid_df$x6 = sqrt(valid_df$swti) 
  # valid_df$x7 = log(valid_df$cwti) 
  # valid_df$x8 = (valid_df$ei)^(-1) 
  
  test_df$x5 = (test_df$vwti)^2
  test_df$x6 = sqrt(test_df$swti) 
  test_df$x7 = log(test_df$cwti) 
  test_df$x8 = (test_df$ei)^(-1) 
  
  train_val_y = as.data.frame(train_df[-5])
  train_val_x = train_df$class

  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    model_rlr_cv = cv.glmnet(x = as.matrix(train[c(-5)]),
                             nfolds = 3,
                             y = train$class,
                             family = "binomial",
                             type.measure = "class",
                             alpha = 0
                             )
    
    model_rlr = glmnet(x = as.matrix(train[(-5)]), 
                       y = train$class,
                       family = "binomial",
                       alpha = 0,
                       lambda = model_rlr_cv$lambda.1se)
    
    
    rlr_predict = predict(model_rlr,
                          newx = as.matrix(validation[(-5)]),
                          type = "response")
    
    rlr_predict = ifelse(rlr_predict < 0.5, 0, 1)
    rlrMetrics = getMetrics(as.numeric(rlr_predict), validation$class)
    resultMatrixCV[i,] = rlrMetrics
    
    #View(resultMatrixCV)
    if(rlrMetrics[1] < bestModelAcc){
      bestModelAcc = rlrMetrics[1]
      bestModelIndex = i
      bestModel = model_rlr_cv
    }
  }
  
  write.csv(resultMatrixCV, file = "Results/rlr_cv_metrics.csv")
  
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
  
  icRLR = as.data.frame(matrix(nrow = 4, ncol=2))
  names(icRLR) = c('Inf', 'Sup')
  icRLR[1,] = icErroRate
  icRLR[2,] = icPrecision
  icRLR[3,] = icRecall
  icRLR[4,] = icf1Score
  
  write.csv(icRLR, file = "Results/rlr_ic.csv")
  
  # Treinamento
  rlr_predict_train = predict(model_rlr_cv,
                              newx = as.matrix(train_df[(-5)]),
                              type = "response")
  
  rlr_predict_train = ifelse(rlr_predict_train < 0.5, 0, 1)
  rlrMetrics_train = getMetrics(rlr_predict_train, train_df$class)
  
  # Teste
  rlr_predict_test = predict(model_rlr_cv,
                             newx = as.matrix(test_df[(-5)]),
                             type = "response")
  
  rlr_predict_test = ifelse(rlr_predict_test < 0.5, 0, 1)
  rlrMetrics_test = getMetrics(rlr_predict_test, test_df$class)
  
  
  # IC para teste
  resultIC = as.data.frame(matrix(ncol=4, nrow=3))
  names(resultIC) = c('ErroRate', 'Precision', 'recall', 'F1')
  ##rownames(resultIC) = c('Métrica', 'Sup.', 'Inf.')
  resultIC$ErroRate[1] = rlrMetrics[1]
  resultIC$ErroRate[2] = rlrMetrics$ErroRate - (2.262*sdErroRateresult)/sqrt(10)
  resultIC$ErroRate[3] = rlrMetrics$ErroRate + (2.262*sdErroRateresult)/sqrt(10)
  resultIC$Precision[1] = rlrMetrics[2]
  resultIC$Precision[2] = rlrMetrics$Precision - (2.262*sdPrecision)/sqrt(10)
  resultIC$Precision[3] = rlrMetrics$Precision + (2.262*sdPrecision)/sqrt(10)
  resultIC$recall[1] = rlrMetrics[3]
  resultIC$recall[2] = rlrMetrics$recall - (2.262*sdRecall)/sqrt(10)
  resultIC$recall[3] = rlrMetrics$recall + (2.262*sdRecall)/sqrt(10)
  resultIC$F1[1] = rlrMetrics[4]
  resultIC$F1[2] = rlrMetrics$F1 - (2.262*sdRecall)/sqrt(10)
  resultIC$F1[3] = rlrMetrics$F1 + (2.262*sdRecall)/sqrt(10)
  
  if(exportResults == T){
    write.csv(rlrMetrics, file = "Results/rlr_metrics.csv")
    #write.csv(resultIC, file = "Results/cbg_resultIC.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(test_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'RLR')
  result_df$Target = test_df$class
  result_df$RLR = rlr_predict_test
  
  return(list('Model'= bestModel,
              'Metrics'= rlrMetrics_test,
              'Results'= result_df)
  )
}

# M6 - Ensemble com regra do voto majoritario (EVM)
getEVM = function(classResult, exportResults = F){
  
  classResult_evm = as.matrix(classResult)
  
  for (i in 1:length(dataNormTest$class)){ #i=1
    nVote = sum(as.numeric(classResult_evm[i, 2:6]))
    
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
