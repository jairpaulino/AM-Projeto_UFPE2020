library(modeest)
library(ks)
x <- rlnorm(10000, meanlog = 3.4, sdlog = 0.2) 


set.seed(123)
# X <- dataNormTrain[c(-5)]
# X.gr <- dataNormTrain$class
X <- dataNormValid[,1:3]
X.gr <- dataNormValid$class
Y <- dataNormTest[,1:3]
Y.gr <- dataNormTest$class
kda.gr <- kda(X, X.gr)
Y.gr.est <- predict(kda.gr, x=Y)
compare(Y.gr, Y.gr.est)

test = compare.kda.diag.cv(X, X.gr, 
                    bw="plugin", 
                    verbose=T)

H <- diag(c(1.25, 0.75, 0.75, 0.75))

# ## univariate example
# ir <- iris[,1]
# ir.gr <- iris[,5]
# kda.fhat <- kda(x=ir, x.group=ir.gr, xmin=3, xmax=9)
# plot(kda.fhat, xlab="Sepal length")
# 
# ## bivariate example
# ir <- iris[,1:2]
# ir.gr <- iris[,5]
# kda.fhat <- kda(x=ir, x.group=ir.gr)
# plot(kda.fhat)
# 
# ## trivariate example
# ir <- iris[,1:3]
# ir.gr <- iris[,5]
# kda.fhat <- kda(x=ir, x.group=ir.gr)
# plot(kda.fhat, drawpoints=TRUE, col.pt=c(2,3,4))
# ## colour=species, transparency=density heights

## univariate example
ir <- dataNormValid[,1]
ir.gr <- as.factor(dataNormValid[,5])
kda.fhat <- kda(x=ir, x.group=ir.gr)
plot(kda.fhat)

## bivariate example
ir <- dataNormValid[,1:2]
ir.gr <- dataNormValid[,5]
kda.fhat <- kda(x=ir, x.group=ir.gr)
plot(kda.fhat)

## trivariate example
ir <- dataNormValid[,1:3]
ir.gr <- dataNormValid[,5]
kda.fhat <- kda(x=ir, x.group=ir.gr)
plot(kda.fhat, drawpoints=TRUE, col.pt=c(2,3,4))

## 4 example
ir <- dataNormValid[,1:4]
ir.gr <- dataNormValid[,5]
kda.fhat <- kda(x=ir, x.group=ir.gr)

## 4 example
fhat <- kde(x=dataNormValid[,1:4])
#plot(fhat, drawpoints=TRUE)
Y.gr.est <- predict(fhat, x = dataNormTest[c(-5)])
Y.gr.est


f_data = 762+610
f_0 = 762/f_data
f_1 = 610/f_data

zero_X = dataNorm[1:762, 1:4]
zero_y = dataNorm[1:762, 5]

hum_X = dataNorm[763:1372, 1:4]
hum_y = dataNorm[763:1372, 5]

h = 0.1
H <- diag(c(h, h, h, h))

zeroModel = kde(x=zero_X[,1:4], H = H)
zeroModel_prob = zeroModel$estimate*f_0

humModel = kde(x=hum_X[,1:4], H = H)
humModel_prob = humModel*f_1

length(weights)

#1, 100, 763, 1000


ParzenClf = NULL

for(i in 1:length(dataNorm$class)){
  prob_0 = predict(zeroModel, x = dataNorm[i,1:4])
  prob_ap_0 = prob_0*f_0
  prob_1 = predict(humModel, x = dataNorm[i,1:4])
  prob_ap_1 = prob_1*f_1
  
  if(prob_0 > prob_1){
    ParzenClf[i] = 0
  }else{
    ParzenClf[i] = 1
  }  
}

getMetrics(ParzenClf, dataNorm$class)


################



count_0 = sum(dataNormTrain[,5] == 1)
count_1 = sum(dataNormTrain[,5] == 0)

count_total = count_0+count_1
f_0 = count_0/f_data
f_1 = count_1/f_data

sample = dataNormTrain[,5] == 0
data_0 = subset(dataNormTrain, sample == T)
data_1 = subset(dataNormTrain, sample == F)

zero_X = data_0[, 1:4]
zero_y = data_0[, 5]

hum_X = data_1[, 1:4]
hum_y = data_1[, 5]

h = 0.01
H = diag(c(h, h, h, h))

zeroModel = kde(x=zero_X[,1:4], H = H)
#zeroModel_prob = zeroModel$estimate*f_0

humModel = kde(x=hum_X[,1:4], H = H)
#humModel_prob = humModel*f_1

ParzenClf = NULL

for(i in 1:length(dataNormTrain$class)){#i=1
  prob_0 = predict(zeroModel, x = dataNormTrain[i, 1:4])
  prob_ap_0 = prob_0*f_0
  prob_1 = predict(humModel, x = dataNormTrain[i, 1:4])
  prob_ap_1 = prob_1*f_1
  
  if(prob_0 > prob_1){
    ParzenClf_train[i] = 0
  }else{
    ParzenClf_train[i] = 1
  }  
  
}

getMetrics(ParzenClf_train, dataNorm$class)
parzenClf_test = NULL
for(i in 1:length(dataNormTest$class)){#i=1
  prob_0 = predict(zeroModel, x = dataNormTest[i, 1:4])
  prob_ap_0 = prob_0*f_0
  prob_1 = predict(humModel, x = dataNormTest[i, 1:4])
  prob_ap_1 = prob_1*f_1
  
  if(prob_0 > prob_1){
    parzenClf_test[i] = 0
  }else{
    parzenClf_test[i] = 1
  }  
}

getMetrics(parzenClf_test, dataNormTest$class)

