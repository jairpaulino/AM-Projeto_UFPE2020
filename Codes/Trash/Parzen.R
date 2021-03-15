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
      
      print(c('Iter.:', j, h[j], metricValid[4]))
      
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
  result_df$Parzen = parzenMetrics_test
  
  return(list('h'= bestH,
              'Metrics'= parzenMetrics_test,
              'Results'= result_df
              )
         )

}
