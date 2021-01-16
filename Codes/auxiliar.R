panel.hist <- function(x, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

panel.lm <- function (x, y, col = par("col"), bg = NA, pch = par("pch"), 
                      cex = 1, col.line="red") {
  points(x, y, pch = pch, col = col, bg = bg, cex = cex)
  ok <- is.finite(x) & is.finite(y)
  if (any(ok)) {
    abline(lm(y[ok]~x[ok]), col = col.line)
  }
}

normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

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
  erroRate = 1 - acc
  recall = Recall(y_pred, y_true)
  f1Score = F1_Score(y_pred, y_true)
  precision = Precision(y_pred, y_true)
  
  metrics_df = as.data.frame(matrix(ncol=4, nrow=1))           
  names(metrics_df) = c('ErroRate', 'Precision', 'recall', 'F1') 
  metrics_df$ErroRate = erroRate
  metrics_df$Precision = precision
  metrics_df$recall = recall
  metrics_df$F1 = f1Score
  
  return(metrics_df)
}

