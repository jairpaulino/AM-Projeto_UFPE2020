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
library(cvTools)   #cv tools

# Importar funcoes implementadas
source("Codes/auxiliar.R")
source("Codes/modelsAM_cv.R")

# FASE 1 - Pre-processamento ----
#
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
data$class = factor(data$class, levels = c(0, 1))
#head(data); View(data); str(data)

# Normalizar dados
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize_2))
dataNorm$class = data$class
#View(dataNorm)

# Separar em conjunto de validacao (20%)
set.seed(1123)
sample = sample.split(dataNorm$vwti, SplitRatio = 0.8)
dataValid = subset(dataNorm, sample == F)
#dataValid %>% count(class)  
#
# FASE 2 - Modelagem ----
#
# M1 - Classificador bayesiano gaussiano (CBG)
modelCGB = getCBG_cv(train_df = dataNorm,
                     exportResults = TRUE)
# Resultados CBG 
modelCGB$Metrics
modelCGB$IC
modelCGB$Results

# M2 - CBG baseado em K-vizinhos (CBG-kNN)
modelKNN = getKNN_cv(train_df = dataNorm, 
                     valid_df = dataValid,
                     exportResults = T)
# Resultados KNN 
modelKNN$Metrics
modelKNN$IC
modelKNN$Results

# M3 - CBG baseado em Janela de Parzen (CBG-JP)
modelParzen = getParzen_cv(train_df = dataNorm, 
                           valid_df = dataValid,
                           exportResults = T)
# Resultados RL 
modelParzen$BestH
modelParzen$Metrics
modelParzen$IC
modelParzen$Results

# M4 - Regressão Logística (RL)
modelLR = getRL_cv(train = dataNorm,
                   exportResults = T)

# Resultados RL 
modelLR$Metrics
modelLR$IC
modelLR$Results

# M5 - Regressão logistica com Regularizacao (RLR) -
modelRLR = getRLR_cv(train = dataNorm,
                    valid_df = dataValid,
                    exportResults = T)
# Resultados KNN 
modelRLR$Metrics
modelRLR$IC
modelRLR$Results

# M6 - Ensemble com regra do voto majoritario (EVM) 
classResult = data.frame(matrix(ncol=6, nrow=length(dataNorm$class)))
colnames(classResult) = c('Class','CBG', 'KNN', 'Parzen', 'LR', 'RLR')
classResult$Class = dataNorm$class
classResult$CBG = modelCGB$Results[,2]
classResult$KNN = modelKNN$Results[,2]
classResult$Parzen = modelParzen$Results[,2]
classResult$LR = modelLR$Results[,2]
classResult$RLR = modelRLR$Results[,2]
#View(classResult)

modelEVM = getEVM(classResult, exportResults = T)

# Resultados KNN 
modelEVM$Metrics
modelEVM$IC
modelEVM$Results

write.csv(classResult, file = "Results/classResult.csv")

# FASE 3 - Analise dos resultados ----
metricTable = data.frame(matrix(ncol=4, nrow=6))
colnames(metricTable) = c('ErroRate', 'Precision', 'recall', 'F1')
rownames(metricTable) = c('CBG', 'KNN', 'Parzen', 'LR', 'LRR', 'EVM')
metricTable[1,1:4] = colMeans(modelCGB$Metrics)
metricTable[2,1:4] = colMeans(modelKNN$Metrics)
metricTable[3,1:4] = colMeans(modelParzen$Metrics)
metricTable[4,1:4] = colMeans(modelLR$Metrics)
metricTable[5,1:4] = colMeans(modelRLR$Metrics)
metricTable[6,1:4] = colMeans(modelEVM$Metrics)
View(metricTable)

write.csv(metricTable, file = "Results/metricTable.csv")

metricTable$ID = seq.int(nrow(metricTable))
myData = melt(metricTable, id.vars = "ID")
fTeste = friedman.test(myData$value, myData$variable, myData$ID)
fTeste
write.csv(c(fTeste$statistic, fTeste$p.value), file = "Results/fTeste.txt")
