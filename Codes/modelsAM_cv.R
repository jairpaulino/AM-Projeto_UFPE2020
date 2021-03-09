# M1_cv - Classificador bayesiano gaussiano (CBG) com CV
getCBG_cv = function(train_df, exportResults = F){
  #train_df = dataNorm; View(train_df)
  
  set.seed(2311)
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  classAll = 1; bestModelAcc = 100
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    #View(train); View(validation)
    model_cbg_cv = naiveBayes(x = train[-5], 
                              y = train$class)
    
    cgb_predict = predict(model_cbg_cv, validation[c(1:4)])
    cbgMetrics_cv = getMetrics(cgb_predict, validation$class)
    resultMatrixCV[i,] = cbgMetrics_cv
    
    # Classificacao para todos o conjunto de dados
    cgb_predict_all = predict(model_cbg_cv, train_df[c(1:4)])
    cbgMetrics_all = getMetrics(cgb_predict_all, train_df$class)

    #View(resultMatrixCV)
    if(cbgMetrics_cv[1] < bestModelAcc){
      bestModelAcc = cbgMetrics_cv[1]
      classAll = cgb_predict_all
      #bestModel = model_cbg_cv
    }
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/cbg_cv_metrics.csv")
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
  
  icCBG = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icCBG) = c('Mean', 'Inf', 'Sup')
  rownames(icCBG) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icCBG[1,1] = meanErroRateresult
  icCBG[2,1] = meanPrecision
  icCBG[3,1] = meanRecall
  icCBG[4,1] = meanf1Score
  icCBG[1,2:3] = icErroRate
  icCBG[2,2:3] = icPrecision
  icCBG[3,2:3] = icRecall
  icCBG[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icCBG, file = "Results/cbg_ic.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(train_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'CBG')
  result_df$Target = train_df$class
  result_df$CBG = classAll
  
  return(list('Metrics'= resultMatrixCV,
              'IC'= icCBG,
              'Results'= result_df
  )
  )
}

# M2_cv - Classificador bayesiano - KNN com CV
getKNN_cv = function(train_df, valid_df, exportResults = F){
  #train_df = dataNorm; valid_df = dataValid; 
  
  maxF1 = 0; maxk = NULL
  for(k in 1:30){#k=2
    set.seed(1)
    previsoes = knn(train = dataValid[-5], 
                    test = dataValid[-5],
                    cl = dataValid$class,
                    k = k)
    
    metricFI = getMetrics(previsoes, dataValid$class)$F1 
    
    if(metricFI > maxF1){
      maxF1 = metricFI
      maxk = k
    }
  }
  
  set.seed(2311)
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    knn_Predict = knn(train = train[-5], 
                    test = test[-5],
                    cl = train$class,
                    k = maxk)
    
    resultMatrixCV[i,] = getMetrics(knn_Predict, test$class)

    knnMetrics = getMetrics(knn_Predict, test$class)
    resultMatrixCV[i,] = knnMetrics
    
    #View(resultMatrixCV)
    # if(knnMetrics[1] < bestModelAcc){
    #   bestModelAcc = knnMetrics[1]
    #   bestModelIndex = i
    #   }
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/knn_cv_metrics.csv")
  }
  
  classAll = knn(train = train_df[-5], 
                    test = train_df[-5],
                    cl = train_df$class,
                    k = maxk)

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
  
  icKnn = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icKnn) = c('Mean', 'Inf', 'Sup')
  rownames(icKnn) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icKnn[1,1] = meanErroRateresult
  icKnn[2,1] = meanPrecision
  icKnn[3,1] = meanRecall
  icKnn[4,1] = meanf1Score
  icKnn[1,2:3] = icErroRate
  icKnn[2,2:3] = icPrecision
  #icKnn[3,2:3] = icRecall
  icKnn[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icKnn, file = "Results/knn_ic.csv")
  }
  
  result_df = as.data.frame(matrix(nrow = length(train_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'KNN')
  result_df$Target = train_df$class
  result_df$KNN = classAll
  
  return(list('Metrics'= resultMatrixCV,
              'IC'= icKnn,
              'Results'= result_df
         )
  )
  
}

# M3 - CBG baseado em Janela de Parzen (CBG-JP) 
getParzen_cv = function(train_df, valid_df, exportResults = T ) {
  #train_df = dataNorm; valid_df = dataValid
  
  # Validacao para encontrar o h
  count_0 = sum(valid_df[,5] == 1)
  count_1 = sum(valid_df[,5] == 0)
  
  count_total = count_0+count_1
  f_0 = count_0/count_total
  f_1 = count_1/count_total
  
  sample = valid_df[,5] == 0
  data_0 = subset(valid_df, sample == T)
  data_1 = subset(valid_df, sample == F)
  
  zero_X = data_0[, 1:4]
  zero_y = data_0[, 5]
  
  hum_X = data_1[, 1:4]
  hum_y = data_1[, 5]
  
  h = c(0.01, 0.1, 0.5, 1, 1.25); 
  bestH = 0; bestMetric = 0
  for (i in 1:5){#i=1
    
    H <- diag(c(h[i], h[i], h[i], h[i]))
    zeroModel = kde(x=zero_X[,1:4], H = H)
    
    humModel = kde(x=hum_X[,1:4], H = H)
    
    ParzenClf_valid = NULL
    for(k in 1:length(valid_df$class)){#k=1
      
      prob_0 = predict(zeroModel, x = valid_df[k, 1:4])
      prob_ap_0 = prob_0*f_0
      
      prob_1 = predict(humModel, x = valid_df[k, 1:4])
      prob_ap_1 = prob_1*f_1
      
      if(prob_0 > prob_1){
        ParzenClf_valid[k] = 0
      }else{
        ParzenClf_valid[k] = 1
      }  
    }
    
    metricValid = getMetrics(ParzenClf_valid, valid_df$class)
    
    print(paste('Iter.:', i, h[i], metricValid[4], sep = ' '))
    
    if(metricValid[4] > bestMetric){
      bestMetric = metricValid[4] 
      bestH = i
    }
  }
  
  # Validacao cruzada 10-folds
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){#i=3
    
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    
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
    
    H = diag(c(h[bestH], h[bestH], h[bestH], h[bestH]))
    zeroModel = kde(x=zero_X[,1:4], H = H)
    humModel = kde(x=hum_X[,1:4], H = H)
      
    ParzenClf_valid = NULL
    for(k in 1:length(test$class)){#k=1
        
      prob_0 = predict(zeroModel, x = test[k, 1:4])
      prob_ap_0 = prob_0*f_0
        
      prob_1 = predict(humModel, x = test[k, 1:4])
      prob_ap_1 = prob_1*f_1
        
      if(prob_0 > prob_1){
        ParzenClf_valid[k] = 0
      }else{
        ParzenClf_valid[k] = 1
      }  
    }
    
    metricValid = getMetrics(ParzenClf_valid, test$class)
    resultMatrixCV[i, ] = metricValid
    print(paste('Iter.:', i, metricValid[4], sep = ' '))
      
    # if(metricValid[4] > bestMetric){
    #     bestMetric = metricValid[4] 
    #     bestH = j
    # }
  }
    
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/parzen_cv_metrics.csv")
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
  
  icParzen = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icParzen) = c('Mean', 'Inf', 'Sup')
  rownames(icParzen) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icParzen[1,1] = meanErroRateresult
  icParzen[2,1] = meanPrecision
  icParzen[3,1] = meanRecall
  icParzen[4,1] = meanf1Score
  icParzen[1,2:3] = icErroRate
  icParzen[2,2:3] = icPrecision
  icParzen[3,2:3] = icRecall
  icParzen[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icParzen, file = "Results/parzen_ic.csv")
  }
  
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
  
  H = diag(c(bestH, bestH, bestH, bestH))
  zeroModel = kde(x=zero_X_train[,1:4], H = H)
  humModel = kde(x=hum_X_train[,1:4], H = H)
  
  classAll = NULL
  for(i in 1:length(train_df$class)){#i=1
    prob_0 = predict(zeroModel, x = train_df[i, 1:4])
    prob_ap_0 = prob_0*f_0
    prob_1 = predict(humModel, x = train_df[i, 1:4])
    prob_ap_1 = prob_1*f_1
    
    if(prob_0 > prob_1){
      classAll[i] = 0
    }else{
      classAll[i] = 1
    }  
  }
  
  #matriz_parzen = table(train_df$class, parzenClf_train)
  parzenMetrics_train = getMetrics(classAll, train_df$class)
  
  result_df = as.data.frame(matrix(nrow = length(train_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'Parzen')
  result_df$Target = train_df$class
  result_df$Parzen = classAll
  
  return(list('Metrics'= resultMatrixCV,
              'IC'= icParzen,
              'Results'= result_df,
              'BestH' = h[bestH]
              )
         )
}

# M4 - Regressão Logística (RL)
getRL_cv = function(train_df, exportResults = F){
  #train_df = dataNorm; 
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelIndex = 1; bestModelAcc = 100; classAll = NULL
  
  classAll = NULL
  for(i in 1:10){#i=1
    set.seed(2311)
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    #View(train); View(validation)
    
    model_lr_cv = glm(class ~ .,
                      data = train,
                      family = binomial(link = 'logit')
    )
    
    
    lr_predict = predict(model_lr_cv, newdata = test[-5])
    lr_predict = ifelse(lr_predict < 0.5, 0, 1)
    lrMetrics = getMetrics(as.numeric(lr_predict), test$class)
    resultMatrixCV[i,] = lrMetrics
    
    # Classificacao para todos o conjunto de dados
    classAll_i = predict(model_lr_cv, train_df[c(1:4)])
    
    if(lrMetrics[1] < bestModelAcc){
      bestModelAcc = lrMetrics[1]
      classAll = ifelse(classAll_i < 0.5, 0, 1)
    }
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/lr_cv_metrics.csv")
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
  
  icLR = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icLR) = c('Mean', 'Inf', 'Sup')
  rownames(icLR) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icLR[1,1] = meanErroRateresult
  icLR[2,1] = meanPrecision
  icLR[3,1] = meanRecall
  icLR[4,1] = meanf1Score
  icLR[1,2:3] = icErroRate
  icLR[2,2:3] = icPrecision
  icLR[3,2:3] = icRecall
  icLR[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icLR, file = "Results/lr_ic.csv")
  }

  result_df = as.data.frame(matrix(nrow = length(train_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'LR')
  result_df$Target = train_df$class
  result_df$LR = classAll
  
  return(list('Metrics' = resultMatrixCV,
              'IC' = icLR,
              'Results'= result_df)
  )
}

# M5 - Regressão logistica com Regularizacao (RLR) 
getRLR_cv= function(train_df, valid_df, exportResults = F){
  #train_df = dataNorm; valid_df = dataValid

  train_df$x5 = (train_df$vwti)^2
  train_df$x6 = sqrt(train_df$swti) 
  train_df$x7 = log(train_df$cwti) 
  train_df$x8 = (train_df$ei)^(-1) 
  
  valid_df$x5 = (valid_df$vwti)^2
  valid_df$x6 = sqrt(valid_df$swti)
  valid_df$x7 = log(valid_df$cwti)
  valid_df$x8 = (valid_df$ei)^(-1)
 
  train_val_y = as.data.frame(valid_df[-5])
  train_val_x = valid_df$class
  
  model_rlr_cv = cv.glmnet(x = as.matrix(train_val_y[c(-5)]),
                           nfolds = 3,
                           y = train_val_x,
                           family = "binomial",
                           type.measure = "class",
                           alpha = 0
  )
  
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  bestModelAcc = 100; classAll = NULL
  for(i in 1:10){#i=1
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    model_rlr = glmnet(x = as.matrix(train[(-5)]), 
                       y = train$class,
                       family = "binomial",
                       alpha = 0,
                       lambda = model_rlr_cv$lambda.1se)
    
    
    rlr_predict = predict(model_rlr,
                          newx = as.matrix(test[(-5)]),
                          type = "response")
    
    rlr_predict = ifelse(rlr_predict < 0.5, 0, 1)
    rlrMetrics = getMetrics(as.numeric(rlr_predict), test$class)
    resultMatrixCV[i,] = rlrMetrics
    
    # Classificacao para todos o conjunto de dados
    classAll_i = predict(model_rlr, 
                         newx = as.matrix(train_df[(-5)]),
                         type = "response")
    
    #View(resultMatrixCV)
    if(rlrMetrics[1] < bestModelAcc){
      bestModelAcc = rlrMetrics[1]
      classAll = ifelse(classAll_i < 0.5, 0, 1)
    }
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/rlr_cv_metrics.csv")
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

  icRLR = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icRLR) = c('Mean', 'Inf', 'Sup')
  rownames(icRLR) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icRLR[1,1] = meanErroRateresult
  icRLR[2,1] = meanPrecision
  icRLR[3,1] = meanRecall
  icRLR[4,1] = meanf1Score
  icRLR[1,2:3] = icErroRate
  icRLR[2,2:3] = icPrecision
  icRLR[3,2:3] = icRecall
  icRLR[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icRLR, file = "Results/rlr_ic.csv")
  }

  result_df = as.data.frame(matrix(nrow = length(train_df[[1]]), ncol=2))
  names(result_df) = c('Target', 'RLR')
  result_df$Target = train_df$class
  result_df$RLR = classAll
  
  return(list('Metrics' = resultMatrixCV,
              'IC' = icRLR,
              'Results'= result_df)
  )
  
  
}

# M6 - Ensemble com regra do voto majoritario (EVM)
getEVM = function(classResult, exportResults = F){
  
  set.seed(2311)
  foldIndex = cvFolds(length(classResult$Class), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('ErroRate', 'Precision', 'recall', 'F1')
  
  classAll = 1
  for(i in 1:10){#i=1
    test = classResult[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    classResult_evm = as.matrix(test)
    #View(classResult_evm)
    
    evmFold = NULL
    for (j in 1:length(test$Class)){#j=2
      nVote = sum(as.numeric(classResult_evm[j, 2:6]))
      
      if (nVote > 2){
        evmFold[j] = 1
      } else {
        evmFold[j] = 0
      }
      
    }#length(evmFold)
    
    evmMetrics_cv = getMetrics(evmFold, test$Class)
    resultMatrixCV[i,] = evmMetrics_cv
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/evm_metrics.csv")
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
  
  icEVM = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icEVM) = c('Mean', 'Inf', 'Sup')
  rownames(icEVM) = c('ErroRate', 'Precision', 'Recall', 'F1')
  icEVM[1,1] = meanErroRateresult
  icEVM[2,1] = meanPrecision
  icEVM[3,1] = meanRecall
  icEVM[4,1] = meanf1Score
  icEVM[1,2:3] = icErroRate
  icEVM[2,2:3] = icPrecision
  icEVM[3,2:3] = icRecall
  icEVM[4,2:3] = icf1Score
  
  if(exportResults == T){
    write.csv(icEVM, file = "Results/evm_ic.csv")
  }
  
  classResult_evm = as.matrix(classResult)

  for (i in 1:length(classResult$Class)){ #i=1
    nVote = sum(as.numeric(classResult_evm[i, 2:6]))

    if (nVote > 2){
      classResult$EVM[i] = 1
    } else {
      classResult$EVM[i] = 0
    }
    
  } #View(classResult)
  
  matriz_evm = table(classResult$Class, classResult$EVM)
  evm_regMetrics = getMetrics(classResult$EVM, classResult$Class)
  
  result_df = as.data.frame(matrix(nrow = length(classResult$EVM), ncol=2))
  names(result_df) = c('Target', 'EVM')
  result_df$Target = classResult$Class
  result_df$EVM = classResult

  return(list('Metrics'= resultMatrixCV,
              'IC' = icEVM,
              'Results'= result_df)
  )
}
