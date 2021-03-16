# Normalize functions
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

getMetrics = function(y_pred, y_true){
  #y_pred = pred_RL_reg; y_true = lr_reg_test_dep
  
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

