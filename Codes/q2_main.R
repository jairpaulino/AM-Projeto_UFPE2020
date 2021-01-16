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

# Chamar funcoes
source("Codes/auxiliar.R")

# Fase 1 - Pre-processamento ----
# Importar dados
data = read.csv("Data/data_banknote_authentication.csv", sep = ";")
#head(data); View(data)
data$class = factor(data$class, levels = c(0, 1))

# Descricao dos dados
summary(data)
 
# ggpairs(data[,1:4], lower = list(continuous = "smooth"))
ggpairs(data, columns = 1:4, ggplot2::aes(colour=class))
ggcorr(data, label=T)

# M <- cor(data[1:4])
# corrplot(M, method = "circle", type = "upper")

# Normalizar
dataNorm = as.data.frame(sapply(data[,1:4], FUN = normalize_2))
dataNorm$class = data$class

# Separar em conjunto de treinamento e teste
sample = sample.split(dataNorm$vwti, SplitRatio = 0.8)
dataNormTrain = subset(dataNorm, sample == T)
dataNormTest = subset(dataNorm, sample == F)

# Fase 2 - Modelagem ----

# M1 - Classificador bayesiano gaussiano (CBG) ----
CBG = naiveBayes(x = dataNormTrain[-5], y = dataNormTrain$class)
print(CBG)

pred_CBG = predict(CBG, newdata = dataNormTest[-5])

matriz_CBG = table(dataNormTest$class, pred_CBG)

confusionMatrix(matriz_CBG, positive = "1")

cbgMetrics = getMetrics(pred_CBG, dataNormTest$class)
write.csv(cbgMetrics, file = "Results/metrics_cbg.csv")

# M2 - Classificador bayesiano gaussiano baseado em K-vizinhos (CB-kV)
# M3 - Classificador bayesiano gaussiano baseado em Janela de Parzen (CB-JP). 
# M4 - Regressão Logística (RL) ----

RL = glm(class ~ ., 
         data = dataNormTrain,
         family = binomial(link = 'logit')
         )

summary(RL)
stargazer(RL, title = "Resultados", type = "text")

pred_RL = predict(RL, newdata = dataNormTest[-5], type = "response")
#View(pred_RL)
plot(pred_RL)

# analises extras 
step(RL, direction = "both") #stepWise
vif(RL) #VIF

pred_RL = ifelse(pred_RL < 0.5, 0, 1)

matriz_RL = table(dataNormTest$class, pred_RL)

confusionMatrix(matriz_RL, positive = "1")

rlMetrics = getMetrics(pred_RL, dataNormTest$class)
write.csv(rlMetrics, file = "Results/metrics_rl.csv")

# M5 - Regressão logistica com Regularizacao (RLR) ----
# M6 - Ensemble com regra do voto majoritario (ERVM)

data_lr = dataNorm
data_lr$x5 = (dataNorm$vwti)^2
data_lr$x6 = sqrt(dataNorm$swti) 
data_lr$x7 = log(dataNorm$cwti) 
data_lr$x8 = (dataNorm$ei)^(-1) 

#lr_reg_ind = data_lr[,c(1:4,6:9)]
#lr_reg_dep = data_lr$class

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

cvLR = cv.glmnet(x = as.matrix(train_val_y), #lr_reg_ind
                 y = train_val_x, #lr_reg_dep
                 family = "binomial",
                 type.measure = "deviance", 
                 nfolds = 10,
                 alpha = 1, #lasso
                 parallel = F)

# trainamento

# set.seed(123) 
# sample_val = sample.split(data_lr$vwti, SplitRatio = 0.2)
# sample_val = subset(data_lr, sample == F)

lr_reg_train_ind = train_rlr[,c(1:4,6:9)]
lr_reg_train_dep = train_rlr$class

lr_reg = glmnet(x = as.matrix(lr_reg_train_ind), 
                y = lr_reg_train_dep,
                family = "binomial",
                lambda = cvLR$lambda.1se)
# test

lr_reg_test_ind = test_rlr[,c(1:4,6:9)]
lr_reg_test_dep = test_rlr$class


pred_RL_reg = predict(lr_reg,
                      s = cvLR$lambda.1se,
                      newx = as.matrix(lr_reg_test_ind),
                      type = "response")

pred_RL_reg = ifelse(pred_RL_reg < 0.5, 0, 1)

rl_regMetrics = getMetrics(pred_RL_reg, lr_reg_test_dep)
write.csv(rl_regMetrics, file = "Results/metrics_rl_reg.csv")

# RL_reg = glm(class ~ ., 
#          data = data_lr,
#          family = binomial(link = 'logit')
# )
# 
# summary(RL_reg)
# stargazer(RL_reg, title = "Resultados", type = "text")
# 
# pred_RL_reg = predict(RL_reg, newdata = data_lr[-5], type = "response")
# #View(pred_RL)
# plot(pred_RL_reg)

# analises extras 
# step(RL_reg, direction = "both") #stepWise
# vif(RL_reg) #VIF
# 
# pred_RL_reg = ifelse(pred_RL_reg < 0.5, 0, 1)
# 
# matriz_RL_reg = table(data_lr$class, pred_RL_reg)
# 
# confusionMatrix(matriz_RL_reg, positive = "1")
# 
# rlreg_Metrics = getMetrics(pred_RL_reg, data_lr$class)
# write.csv(rlreg_Metrics, file = "Results/metrics_rl_reg.csv")


# Fase 3 - Teste ----

