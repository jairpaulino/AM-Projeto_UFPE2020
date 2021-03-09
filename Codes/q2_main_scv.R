#Title: Repeated K-Fold CV to classification
#Author: Jair Paulino
#Date: 2021/02/15

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
library(MLmetrics) #metrics
library(glmnet)    #cv.glmnet
library(class)     #knn
library(dplyr)     #dados
library(reshape2)  #dados
library(ks)        #Parzen
library(klaR)
library(cvTools)  #cv tools

# Importar funcoes implementadas
source("Codes/auxiliar.R")
source("Codes/modelsAM_cv.R")

# FASE 1 - Pre-processamento ----
#
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
data$class = factor(data$class, levels = c(0, 1))
#head(data); View(data); str(data)

# Separar em conjunto de treinamento, validacao e teste
# set.seed(123) 
# sample = sample.split(data$vwti, SplitRatio = 0.8)
# dataTrain = subset(data, sample == T)
# dataTest = subset(data, sample == F)
# dataTrain %>% count(class) #str(dataTrain)
# dataTest %>% count(class)  #str(dataTest)

# Normalizar dados
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize_2))
dataNorm$class = data$class
View(dataNorm)

# dataNormTrain = as.data.frame(matrix(ncol=5, nrow=length(dataTrain$vwti)))
# names(dataNormTrain) = names(dataTrain)
# for (i in 1:4){#i=1
#   dataNormTrain[,i] = normalize_minMax(dataTrain[,i], 
#                                        max = max(dataTrain[,i]),
#                                        min = min(dataTrain[,i]))
# } 
# dataNormTrain$class = dataTrain$class #View(dataNormTrain)
# 
# # Normalizar conjunto de treinamento
# dataNormTrain = as.data.frame(matrix(ncol=5, nrow=length(dataTrain$vwti)))
# names(dataNormTrain) = names(dataTrain)
# for (i in 1:4){#i=1
#   dataNormTrain[,i] = normalize_minMax(dataTrain[,i], 
#                                        max = max(dataTrain[,i]),
#                                        min = min(dataTrain[,i]))
# } 
# dataNormTrain$class = dataTrain$class #View(dataNormTrain)
# #class(dataNormTrain)
# 
# # Normalizar conjunto de teste
# dataNormTest = as.data.frame(matrix(ncol=5, nrow=length(dataTest$vwti)))
# names(dataNormTest) = names(dataTest)
# for (i in 1:4) {
#   dataNormTest[,i] = normalize_minMax(dataTest[,i],
#                                       max = max(dataTrain[,i]),
#                                       min = min(dataTrain[,i]))
# }
# dataNormTest$class = dataTest$class #View(dataNormTest)
# #class(dataNormTest)

# FASE 2 - Modelagem ----
# M1 - Classificador bayesiano gaussiano (CBG)
modelCGB = getCBG_cv(train = dataNorm,
                     exportResults = T)
# Resultados CBG 
modelCGB$Metrics
modelCGB$IC
modelCGB$Results

# M2 - CBG baseado em K-vizinhos (CBG-kNN)
modelKNN = getKNN_cv(train_df = dataNormTrain,
                     test_df = dataNormTest,
                     exportResults = T)
# Resultados KNN 
modelKNN$K
modelKNN$Metrics
modelKNN$Results

# M3 - CBG baseado em Janela de Parzen (CBG-JP)

modelParzen = getParzen_cv(train_df = dataNormTrain,
                           test_df = dataNormTest,
                           exportResults = T)
# Resultados RL 
modelParzen$h
modelParzen$Metrics
modelParzen$Results

# M4 - Regressão Logística (RL)
modelLR = getRL_cv(train = dataNormTrain,
                   test = dataNormTest,
                   exportResults = T)

# Resultados RL 
modelLR$Model
modelLR$Metrics
modelLR$Results

# M5 - Regressão logistica com Regularizacao (RLR) -
modelRLR = getRLR_cv(train = dataNormTrain,
                     test = dataNormTest,
                     exportResults = T)
# Resultados KNN 
modelRLR$Model
modelRLR$Metrics
modelRLR$Results

# M6 - Ensemble com regra do voto majoritario (EVM) 
classResult = data.frame(matrix(ncol=6, nrow=length(dataNormTest$class)))
colnames(classResult) = c('Target','CBG', 'KNN', 'Parzen', 'LR', 'RLR')
classResult$Target = dataNormTest$class
classResult$CBG = modelCGB$Results[,2]
classResult$KNN = modelKNN$Results[,2]
classResult$Parzen = modelParzen$Results
classResult$LR = modelLR$Results[,2]
classResult$RLR = modelRLR$Results[,2]
#View(classResult)

modelEVM = getEVM(classResult, exportResults = T)
modelEVM$Metrics
modelEVM$Results

# FASE 3 - Analise dos resultados ----
metricTable = data.frame(matrix(ncol=4, nrow=6))
colnames(metricTable) = c('ErroRate', 'Precision', 'recall', 'F1')
rownames(metricTable) = c('CBG', 'KNN', 'Parzen', 'LR', 'LRR', 'EVM')
metricTable[1,1:4] = modelCGB$Metrics
metricTable[2,1:4] = modelKNN$Metrics
metricTable[3,1:4] = modelParzen$Metrics
metricTable[4,1:4] = modelLR$Metrics
metricTable[5,1:4] = modelRLR$Metrics
metricTable[6,1:4] = modelEVM$Metrics
View(metricTable)

write.csv(metricTable, file = "Results/metricTable.csv")

metricTable$ID = seq.int(nrow(metricTable))
myData = melt(metricTable, id.vars = "ID")
fTeste = friedman.test(myData$value, myData$variable, myData$ID)
fTeste
write.csv(c(fTeste$statistic, fTeste$p.value), file = "Results/fTeste.txt")

