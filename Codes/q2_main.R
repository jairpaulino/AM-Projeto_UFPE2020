# Inicializacao ----
# limpar dados e figuras
rm(list = ls()); graphics.off()

# Bibliotecas
library(GGally)   #ggpairs
library(ggplot2)  #ggplot
library(corrplot) #cor
library(caTools)  #sample.split
library(e1071)    #naiveBayes
library(caret)    #confusionMatrix
 
# Chamar funcoes
source("Codes/auxiliar.R")

# Fase 1 - Pre-processamento ----
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
#head(data); View(data)
data$class = factor(data$class, levels = c(0, 1))

# Descricao dos dados
summary(data)
pairs(data[,1:4], diag.panel = panel.hist)
pairs(data[,1:4], diag.panel = panel.hist, 
      upper.panel = panel.cor)
pairs(data[,1:4], diag.panel = panel.hist, 
      upper.panel = panel.cor,
      lower.panel = panel.lm)

ggpairs(data[,1:4], lower = list(continuous = "smooth"))
ggpairs(data, columns = 1:4, ggplot2::aes(colour=class))
ggcorr(data, label=T)

M <- cor(data[1:4])
corrplot(M, method = "circle", type = "upper")

# Normalizar
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize))
dataNorm$class = data$class

# Separar em conjunto de treinamento e teste
sample = sample.split(dataNorm$vwti, SplitRatio = 0.8)
dataNormTrain = subset(dataNorm, sample == T)
dataNormTest = subset(dataNorm, sample == F)
#perc = 0.8
#dataNormTrain = dataNorm[1:round(perc*length(data$vwti)),]
#dataNormTest = dataNorm[(round(perc*length(data$vwti))+1):length(data$vwti),]

# Fase 2 - Modelagem ----

# M1 - Classificador bayesiano gaussiano (CBG)
CBG = naiveBayes(x = dataNormTrain[-5], y = dataNormTrain$class)
print(CBG)

pred_CBG = predict(CBG, newdata = dataNormTest[-5])

matriz_CBG = table(dataNormTest$class, pred_CBG)

confusionMatrix(matriz_CBG, positive = "1")

# Fase 3 - Teste ----

