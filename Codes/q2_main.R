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

# FASE 1 - Pre-processamento ----
#
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
data$class = factor(data$class, levels = c(0, 1))
#head(data); View(data)

# Normalizar dados
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize_2))
dataNorm$class = data$class

# Separar em conjunto de treinamento, validacao e teste
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

# FASE 2 - Modelagem ----
# M1 - Classificador bayesiano gaussiano (CBG)
modelCGB = getGBG(train = dataNormTrain,
                  test = dataNormTest,
                  exportResults = T)
# Resultados CBG 
modelCGB$Model
modelCGB$Metrics
modelCGB$Results

# M2 - CBG baseado em K-vizinhos (CBG-kNN)
modelKNN = getKNN(train = dataNormTrain,
                  valid = dataNormValid,
                  test = dataNormTest,
                  exportResults = T)
# Resultados KNN 
modelKNN$Model
modelKNN$Metrics
modelKNN$Results

# M3 - CBG baseado em Janela de Parzen (CBG-JP)

# M4 - Regressão Logística (RL)
modelLR = getRL(train = dataNormTrain,
                  test = dataNormTest,
                  exportResults = T)
# Resultados KNN 
modelLR$Model
modelLR$Metrics
modelLR$Results

# M5 - Regressão logistica com Regularizacao (RLR) -
modelLRR = getLRR(train = dataNormTrain,
                  valid = dataNormValid,
                  test = dataNormTest,
                  exportResults = T)
# Resultados KNN 
modelLRR$Model
modelLRR$Metrics
modelLRR$Results

# M6 - Ensemble com regra do voto majoritario (EVM) 
classResult = data.frame(matrix(ncol=5, nrow=length(dataNormTest$class)))
colnames(classResult) = c('Target','CBG', 'KNN', 'LR', 'LRR')
classResult$Target = dataNormTest$class
classResult$CBG = modelCGB$Results[,2]
classResult$KNN = modelKNN$Results[,2]
classResult$LR = modelLR$Results[,2]
classResult$LRR = modelLRR$Results[,2]
#View(classResult)

modelEVM = getEVM(classResult, exportResults = T)
modelEVM$Metrics
modelEVM$Results

# FASE 3 - Analise dos resultados ----
metricTable = data.frame(matrix(ncol=4, nrow=5))
colnames(metricTable) = c('ErroRate', 'Precision', 'recall', 'F1')
rownames(metricTable) = c('CBG', 'KNN', 'LR', 'LRR', 'EVM')
metricTable[1,1:4] = modelCGB$Metrics
metricTable[2,1:4] = modelKNN$Metrics
metricTable[3,1:4] = modelLR$Metrics
metricTable[4,1:4] = modelLRR$Metrics
metricTable[5,1:4] = modelEVM$Metrics
View(metricTable)

write.csv(metricTable, file = "Results/metricTable.csv")

# Finalizar Parzer (em R)
# Calcular tempo de processamento para cada modelo
# Implementar Friedman test
