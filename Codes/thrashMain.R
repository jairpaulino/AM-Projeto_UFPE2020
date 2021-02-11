# Inicializacao ----
# limpar dados e figuras
rm(list = ls()); graphics.off()

# Bibliotecas
library(GGally)    #ggpairs
library(ggplot2)   #ggplot
library(corrplot)  #cor
library(caTools)   #sample.split
library(e1071)     #naiveBayes
library(caret)     #confusionMatrix
library(MLmetrics)
library(stargazer) #stargazer
library(faraway)   #vif
library(glmnet)    #cv.glmnet
library(class)     #knn
library(reticulate)
library(dplyr)

# library(kdensity)  #
# library(ks)        #
# library(Compositional)      #

# Chamar funcoes
source("Codes/auxiliar.R")
source("Codes/modelsAM.R")

# Fase 1 - Pre-processamento ----
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
data$class = factor(data$class, levels = c(0, 1))
#head(data); View(data)

# Normalizar
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize_2))
dataNorm$class = data$class

# Separar em conjunto de treinamento e teste
set.seed(123) 
sample = sample.split(dataNorm$vwti, SplitRatio = 0.8)
dataNormTrain = subset(dataNorm, sample == T)
dataNormTest = subset(dataNorm, sample == F)
#dataNormTrain %>% count(class)
#dataNormTest %>% count(class)

set.seed(123) 
sample_val = sample.split(dataNormTrain$vwti, SplitRatio = 0.2)
dataNormValid = subset(dataNormTrain, sample_val == T)
#dataNormValid %>% count(class)

# Fase 2 - Modelagem ----
# M1 - Classificador bayesiano gaussiano (CBG)
modelCGB = getGBG(train = dataNormTrain,
                  test = dataNormTest,
                  exportResults = T)
# Resultados CBG 
modelCGB$Model
modelCGB$Metrics
modelCGB$Results

# M2 - CBG baseado em K-vizinhos (CBG-kV) ----
modelKNN = getKNN(train = dataNormTrain,
                  valid = dataNormValid,
                  test = dataNormTest,
                  exportResults = T)
# Resultados KNN 
modelKNN$Model
modelKNN$Metrics
modelKNN$Results


# M2 - CBG baseado em K-vizinhos (CBG-kV) ----
# 
# set.seed(123) 
# sample_knn_val = sample.split(dataNormTrain$vwti, SplitRatio = 0.2)
# validation_knn = subset(dataNormTrain, sample_knn_val == T)

bestAcc = 0; kPos = 1; kNNAcu = NULL
for (i in 3:100){#i=1
  
  modelKNN = knn(train = dataNormTrain[,-5],
                 test = dataNormTest[-5],
                 cl = dataNormTrain$class,
                 k = i) 
  
  matriz_knn = table(dataNormTest$class, modelKNN)
  
  cm = confusionMatrix(matriz_knn, positive = "1")
  
  kNNAcu = c(kNNAcu, as.numeric(cm$overall[1]))
  
  if(bestAcc < cm$overall[1]){
    bestAcc = cm$overall[1]
    kPos = i
  }
}

modelKNN = knn(train = dataNormTrain[,-5],
               test = dataNormTest[-5],
               cl = dataNormTrain$class,
               k = kPos) 

summary(modelKNN)

matriz_knn = table(dataNormTest$class, modelKNN); matriz_knn

confusionMatrix(matriz_knn, positive = "1")

knnMetrics = getMetrics(dataNormTest$class, modelKNN)
write.csv(knnMetrics, file = "Results/metrics_knn.csv")

classResult$CBG_kV = modelKNN
#View(classResult)

# M3 - CBG baseado em Janela de Parzen (CBG-JP) -----

require(graphics)


# M4 - Regressão Logística (RL) ----

RL = glm(class ~ ., 
         data = dataNormTrain,
         family = binomial(link = 'logit')
)

summary(RL)
#stargazer(RL, title = "Resultados", type = "text")

pred_RL = predict(RL, 
                  newdata = dataNormTest[-5], 
                  type = "response")
#plot(pred_RL)

# analises extras 
# step(RL, direction = "both") #stepWise
# vif(RL) #VIF

pred_RL = ifelse(pred_RL < 0.5, 0, 1)

matriz_RL = table(dataNormTest$class, pred_RL)

confusionMatrix(matriz_RL, positive = "1")

rlMetrics = getMetrics(pred_RL, dataNormTest$class)
write.csv(rlMetrics, file = "Results/metrics_rl.csv")

classResult$RL = pred_RL
#View(classResult)

# M5 - Regressão logistica com Regularizacao (RLR) ----

data_lr = dataNorm
data_lr$x5 = (dataNorm$vwti)^2
data_lr$x6 = sqrt(dataNorm$swti) 
data_lr$x7 = log(dataNorm$cwti) 
data_lr$x8 = (dataNorm$ei)^(-1) 

# split e validation
set.seed(123) 
sample_rlr = sample.split(data_lr$vwti, SplitRatio = 0.8)
train_rlr = subset(data_lr, sample_rlr == T)
test_rlr = subset(data_lr, sample_rlr == F)

set.seed(123) 
sample_rlr_val = sample.split(train_rlr$vwti, SplitRatio = 0.2)
validation_rlr = subset(train_rlr, sample_rlr_val == T)

train_val_y = as.data.frame(validation_rlr[-5]) 
train_val_x = (validation_rlr$class)

# RLR - cross-validation 
cvLR = cv.glmnet(x = as.matrix(train_val_y), #lr_reg_ind
                 y = train_val_x, #lr_reg_dep
                 family = "binomial",
                 type.measure = "deviance", 
                 nfolds = 10,
                 alpha = 1, #lasso
                 parallel = F)

# Treinamento

lr_reg_train_y = train_rlr[,c(1:4,6:9)]
lr_reg_train_x = train_rlr$class

lr_reg = glmnet(x = as.matrix(lr_reg_train_y), 
                y = lr_reg_train_x,
                family = "binomial",
                lambda = cvLR$lambda.1se)
# Test

lr_reg_test_y = test_rlr[,c(1:4,6:9)]
lr_reg_test_x = test_rlr$class

pred_RLR = predict(lr_reg,
                   s = cvLR$lambda.1se,
                   newx = as.matrix(lr_reg_test_y),
                   type = "response")

pred_RLR = ifelse(pred_RLR < 0.5, 0, 1)

rlr_regMetrics = getMetrics(pred_RLR, lr_reg_test_x)
write.csv(rlr_regMetrics, file = "Results/metrics_rlr.csv")

classResult$RLR = pred_RLR
#View(classResult)

# M6 - Ensemble com regra do voto majoritario (EVM) ----

classResult_evm = as.matrix(classResult)

for (i in 1:length(classResult$target)){ #i=1
  nVote = sum(as.numeric(classResult_evm[i, 2:5]))
  
  if (nVote > 2){
    classResult$EVM[i] = 1
  } else {
    classResult$EVM[i] = 0
  }
  
} #View(classResult)

evm_regMetrics = getMetrics(classResult$EVM, dataNormTest$class)
write.csv(evm_regMetrics, file = "Results/metrics_evm.csv")

length(classResult$EVM)

# Fase 3 - Teste ----

# dd = iris
# head(iris)
# View(iris)
# 
# count(iris$Species)
# iris %>% group_by(Species) %>% count()
# data %>% group_by(class) %>% count()


