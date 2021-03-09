# limpar dados e figuras
rm(list = ls()); graphics.off()

# Bibliotecas
library(ggplot2)   #ggplot
library(caTools)   #sample.split
library(caret)     #confusionMatrix
library(MLmetrics) #metrics
library(dplyr)     #dados
library(h2o)       #H2O


# Importar funcoes implementadas
source("Codes/auxiliar.R")

# FASE 1 - Pre-processamento ----
# Importar dados
data = read.table('C:/Users/jairp/Desktop/glass.data', sep=',')
names(data) = c('Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class')
data$Id = NULL
#head(data); View(data); str(data)

# Separar em conjunto de treinamento, validacao e teste
set.seed(1235) 
sample = sample.split(data$RI, SplitRatio = 0.70)
dataTrain = subset(data, sample == T)
dataTest = subset(data, sample == F)
dataTrain %>% count(Class) #str(dataTrain)
dataTest %>% count(Class)  #str(dataTest)

# Normalizar conjunto de treinamento
dataNormTrain = as.data.frame(matrix(ncol=10, nrow=length(dataTrain$RI)))
names(dataNormTrain) = names(dataTrain)
for (i in 1:9){#i=1
  dataNormTrain[,i] = normalize_minMax(dataTrain[,i], 
                                       max = max(dataTrain[,i]),
                                       min = min(dataTrain[,i]))
} 
dataNormTrain$Class = as.factor(dataTrain$Class)
#View(dataNormTrain)


# Normalizar conjunto de teste
dataNormTest = as.data.frame(matrix(ncol=10, nrow=length(dataTest$RI)))
names(dataNormTest) = names(dataTest)
for (i in 1:9) {
  dataNormTest[,i] = normalize_minMax(dataTest[,i],
                                      max = max(dataTrain[,i]),
                                      min = min(dataTrain[,i]))
}
dataNormTest$Class = as.factor(dataTest$Class)
#View(dataNormTest)

h2o.init(nthreads = -1)
seed = 123
model = h2o.deeplearning(y = "Class",
                         x = names(dataNormTrain)[c(1:9)],
                         keep_cross_validation_models = TRUE, 
                         training_frame = as.h2o(dataNormTrain),
                         activation = 'Maxout', 
                         nfolds = 3,
                         hidden = c(20, 20),
                         epochs = 200, 
                         train_samples_per_iteration = -2,
                         stopping_tolerance = 0.01)

#plot(model)
#h2o.performance(model, train = T)
#h2o.performance(model, newdata = as.h2o(dataNormTest))

forecast_results = as.data.frame(h2o.predict(model, 
                                             newdata = as.h2o(dataNormTest), 
                                             rep = 5, seed = 123))
mTable = table(forecast_results[[1]], dataNormTest$Class)
mTable

#Accuracy(forecast_results[[1]], dataNormTest$Class)
F1_Score(forecast_results[[1]], dataNormTest$Class)
Recall(forecast_results[[1]], dataNormTest$Class)
Precision(forecast_results[[1]], dataNormTest$Class)
