#data = read.csv(file.choose(), sep = ";")

library(quantmod)
library(neuralnet)

df = data.frame(data)
names(df) = 'training'
df['lag1'] = Lag(data,1)
df['lag2'] = Lag(data,2)
df['lag3'] = Lag(data,3)
df['lag4'] = Lag(data,4)
df['lag5'] = Lag(data,5)
df['lag6'] = Lag(data,6)
df = na.omit(df)

h = round(length(df$training)*0.6)
treino = df[1:h,]
teste = df[(h+1):length(df$training),]
completo = df
  
df.treino.n = apply(treino, MARGIN = 2, normalize)
df.teste.n = apply(teste, MARGIN = 2, normalize)
completo_df = rbind(df.treino.n, df.teste.n)
  
set.seed(123)
modelo = neuralnet(training ~ .,
                   data = df.treino.n, 
                   hidden = c(10, 10),
                   act.fct = "logistic",
                   learningrate = 0.05,
                   algorithm = 'backprop',
                   rep = 10,
                   linear.output = F,
                   #threshold = 0.00001
                   )
set.seed(123)
fore = predict(modelo, completo_df, rep = 3)
plot.ts(completo_df[,1], ylim=c(0, 1.4))
lines(fore, col=2, lwd=2)
abline(v=h, lty=3)
legend("topleft", c('TS', 'ANN'), col = c(1,2),
       lty = c(1, 1))


plot.ts(completo_df[h:length(completo_df[,1]),1], ylim=c(0, 1))
lines(fore[h:length(completo_df[,1])], col=2, lwd=2)
abline(v=h, lty=3)
legend("topleft", c('TS', 'ANN'), col = c(1,2),
       lty = c(1, 1))

