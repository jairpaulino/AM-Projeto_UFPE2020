# M1_cv - Classificador bayesiano gaussiano (CBG) com CV
getCBG_cv = function(train_df, exportResults = F){

  set.seed(2311)
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  classAll = 1; bestModelAcc = 100
  for(i in 1:10){
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    validation = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    model_cbg_cv = naiveBayes(x = train[-5], 
                              y = train$class)
    
    cgb_predict = predict(model_cbg_cv, validation[c(1:4)])
    cbgMetrics_cv = getMetrics(cgb_predict, validation$class)
    resultMatrixCV[i,] = cbgMetrics_cv
    
    # Classificacao para todos o conjunto de dados
    cgb_predict_all = predict(model_cbg_cv, train_df[c(1:4)])
    cbgMetrics_all = getMetrics(cgb_predict_all, train_df$class)

    if(cbgMetrics_cv[1] < bestModelAcc){
      bestModelAcc = cbgMetrics_cv[1]
      classAll = cgb_predict_all
    }
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/cbg_cv_metrics.csv")
  }

  # Calculo do IC
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]
  
  icCBG = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icCBG) = c('Mean', 'Inf', 'Sup')
  rownames(icCBG) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icCBG[1,1] = meanTaxaErroresult
  icCBG[2,1] = meanPrecisao
  icCBG[3,1] = meanCobertura
  icCBG[4,1] = meanFmeasureScore
  icCBG[1,2:3] = icTaxaErro
  icCBG[2,2:3] = icPrecisao
  icCBG[3,2:3] = icCobertura
  icCBG[4,2:3] = icFmeasureScore
  
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
  #train_df = dataNorm; valid_df = dataNormValid
  
  maxFmeasure = 0; maxk = NULL
  for(k in 1:30){ #k=1
    set.seed(1)
    previsoes = knn(train = dataValid[-5], 
                    test = dataValid[-5],
                    cl = dataValid$class,
                    k = k)
    
    metricFI = getMetrics(previsoes, dataValid$class)[4] 
    
    if(metricFI > maxFmeasure){
      maxFmeasure = metricFI
      maxk = k
    }
  }
  
  set.seed(2311)
  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

    knn_Predict = knn(train = train[-5], 
                    test = test[-5],
                    cl = train$class,
                    k = maxk)
    
    resultMatrixCV[i,] = getMetrics(knn_Predict, test$class)

    knnMetrics = getMetrics(knn_Predict, test$class)
    resultMatrixCV[i,] = knnMetrics
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/knn_cv_metrics.csv")
  }
  
  classAll = knn(train = train_df[-5], 
                    test = train_df[-5],
                    cl = train_df$class,
                    k = maxk)

  # Calculo do IC
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  #icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]
  
  icKnn = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icKnn) = c('Mean', 'Inf', 'Sup')
  rownames(icKnn) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icKnn[1,1] = meanTaxaErroresult
  icKnn[2,1] = meanPrecisao
  icKnn[3,1] = meanCobertura
  icKnn[4,1] = meanFmeasureScore
  icKnn[1,2:3] = icTaxaErro
  icKnn[2,2:3] = icPrecisao
  #icKnn[3,2:3] = icCobertura
  icKnn[4,2:3] = icFmeasureScore
  
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
  for (i in 1:5){
    
    H <- diag(c(h[i], h[i], h[i], h[i]))
    zeroModel = kde(x=zero_X[,1:4], H = H)
    
    humModel = kde(x=hum_X[,1:4], H = H)
    
    ParzenClf_valid = NULL
    for(k in 1:length(valid_df$class)){
      
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
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  bestModelIndex = 1; bestModelAcc = 100; bestModel = NULL
  for(i in 1:10){
    
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
    for(k in 1:length(test$class)){
        
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
  }
    
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/parzen_cv_metrics.csv")
  }
  
  # Calculo do IC
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]
  
  icParzen = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icParzen) = c('Mean', 'Inf', 'Sup')
  rownames(icParzen) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icParzen[1,1] = meanTaxaErroresult
  icParzen[2,1] = meanPrecisao
  icParzen[3,1] = meanCobertura
  icParzen[4,1] = meanFmeasureScore
  icParzen[1,2:3] = icTaxaErro
  icParzen[2,2:3] = icPrecisao
  icParzen[3,2:3] = icCobertura
  icParzen[4,2:3] = icFmeasureScore
  
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
  for(i in 1:length(train_df$class)){
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

  foldIndex = cvFolds(length(train_df$vwti), K = 10, R = 1)
  resultMatrixCV = as.data.frame(matrix(ncol=4, nrow=10))
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  bestModelIndex = 1; bestModelAcc = 100; classAll = NULL
  
  classAll = NULL
  for(i in 1:10){
    set.seed(2311)
    train = train_df[foldIndex$subsets[foldIndex$which != i], ] #Set the training set
    test = train_df[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set

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
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]
  
  icLR = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icLR) = c('Mean', 'Inf', 'Sup')
  rownames(icLR) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icLR[1,1] = meanTaxaErroresult
  icLR[2,1] = meanPrecisao
  icLR[3,1] = meanCobertura
  icLR[4,1] = meanFmeasureScore
  icLR[1,2:3] = icTaxaErro
  icLR[2,2:3] = icPrecisao
  icLR[3,2:3] = icCobertura
  icLR[4,2:3] = icFmeasureScore
  
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
  #train_df = dataNorm; valid_df = dataNormValid
  
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
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  bestModelAcc = 100; classAll = NULL
  for(i in 1:10){
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
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]

  icRLR = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icRLR) = c('Mean', 'Inf', 'Sup')
  rownames(icRLR) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icRLR[1,1] = meanTaxaErroresult
  icRLR[2,1] = meanPrecisao
  icRLR[3,1] = meanCobertura
  icRLR[4,1] = meanFmeasureScore
  icRLR[1,2:3] = icTaxaErro
  icRLR[2,2:3] = icPrecisao
  icRLR[3,2:3] = icCobertura
  icRLR[4,2:3] = icFmeasureScore
  
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
  names(resultMatrixCV) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  
  classAll = 1
  for(i in 1:10){
    test = classResult[foldIndex$subsets[foldIndex$which == i], ] #Set the validation set
    classResult_evm = as.matrix(test)

    evmFold = NULL
    for (j in 1:length(test$Class)){
      nVote = sum(as.numeric(classResult_evm[j, 2:6]))
      
      if (nVote > 2){
        evmFold[j] = 1
      } else {
        evmFold[j] = 0
      }
      
    }
    
    evmMetrics_cv = getMetrics(evmFold, test$Class)
    resultMatrixCV[i,] = evmMetrics_cv
  }
  
  if(exportResults == T){
    write.csv(resultMatrixCV, file = "Results/evm_metrics.csv")
  }
  
  # Calculo do IC
  meanTaxaErroresult = mean(resultMatrixCV$TaxaErro)
  sdTaxaErroresult = sd(resultMatrixCV$TaxaErro)
  icTaxaErro = (t.test(resultMatrixCV$TaxaErro))$conf.int[1:2]
  
  meanPrecisao = mean(resultMatrixCV$Precisao)
  sdPrecisao = sd(resultMatrixCV$Precisao)
  icPrecisao = (t.test(resultMatrixCV$Precisao))$conf.int[1:2]
  
  meanCobertura = mean(resultMatrixCV$Cobertura)
  sdCobertura = sd(resultMatrixCV$Cobertura)
  icCobertura = (t.test(resultMatrixCV$Cobertura))$conf.int[1:2]
  
  meanFmeasureScore = mean(resultMatrixCV$Fmeasure)
  sdFmeasureScore = sd(resultMatrixCV$Fmeasure)
  icFmeasureScore = (t.test(resultMatrixCV$Fmeasure))$conf.int[1:2]
  
  icEVM = as.data.frame(matrix(nrow = 4, ncol=3))
  names(icEVM) = c('Mean', 'Inf', 'Sup')
  rownames(icEVM) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure')
  icEVM[1,1] = meanTaxaErroresult
  icEVM[2,1] = meanPrecisao
  icEVM[3,1] = meanCobertura
  icEVM[4,1] = meanFmeasureScore
  icEVM[1,2:3] = icTaxaErro
  icEVM[2,2:3] = icPrecisao
  icEVM[3,2:3] = icCobertura
  icEVM[4,2:3] = icFmeasureScore
  
  if(exportResults == T){
    write.csv(icEVM, file = "Results/evm_ic.csv")
  }
  
  classResult_evm = as.matrix(classResult)

  for (i in 1:length(classResult$Class)){
    nVote = sum(as.numeric(classResult_evm[i, 2:6]))

    if (nVote > 2){
      classResult$EVM[i] = 1
    } else {
      classResult$EVM[i] = 0
    }
    
  } 
  
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

# Funcao - Normalize [0.2; 0.8]
normalize_2 = function(array, x = 0.2, y = 0.8){
  #Normalize to [0, 1]
  m = min(array)
  range = max(array) - m
  norm1 = (array - m) / range
  
  #Then scale to [x,y]
  range2 = y - x
  normalized = (norm1*range2) + x
  return(normalized)
}

# Calcula as metricas
getMetrics = function(y_pred, y_true){

  acc = Accuracy(y_pred, y_true)
  TaxaErro = 1 - acc
  recall = Recall(y_pred, y_true)
  f1Score = F1_Score(y_pred, y_true)
  precision = Precision(y_pred, y_true)
  
  metrics_df = as.data.frame(matrix(ncol=4, nrow=1))           
  names(metrics_df) = c('TaxaErro', 'Precisao', 'Cobertura', 'Fmeasure') 
  metrics_df$TaxaErro = TaxaErro
  metrics_df$Precisao = precision
  metrics_df$Cobertura = recall
  metrics_df$Fmeasure = f1Score
  
  return(metrics_df)
}
