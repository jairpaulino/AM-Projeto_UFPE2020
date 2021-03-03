library(modeest)
library(ks)
x <- rlnorm(10000, meanlog = 3.4, sdlog = 0.2) 

## True mode 
lnormMode(meanlog = 3.4, sdlog = 0.2) 

## Estimate of the mode 
mlv(x, method = "parzen", kernel = "gaussian", bw = 0.3, par = shorth(x)) 

set.seed(8192)
x <- c(rnorm.mixt(n=100, mus=1), rnorm.mixt(n=100, mus=-1))
x.gr <- rep(c(1,2), times=c(100,100))
y <- c(rnorm.mixt(n=100, mus=1), rnorm.mixt(n=100, mus=-1))
y.gr <- rep(c(1,2), times=c(100,100))
kda.gr <- kda(x, x.gr)
y.gr.est <- predict(kda.gr, x=y)
compare(y.gr, y.gr.est)

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

## positive data example
fhat <- kde(x=X)

## univariate example
fhat <- kde(x=X[,1])
plot(fhat, cont=50, col.cont="blue", cont.lwd=2, xlab="vwti")

## bivariate example
fhat <- kde(x=X[,1:2])
plot(fhat, display="filled.contour", cont=seq(10,90,by=10))
plot(fhat, display="persp", thin=3, border=1, col="white")

## trivariate example
fhat <- kde(x=X[,1:3])
plot(fhat, drawpoints=TRUE)

## 4 example
fhat <- kde(x=X[,1:4])
plot(fhat, drawpoints=TRUE)
Y.gr.est <- predict(fhat, x = dataNormTest[c(-5)])

zero_X = dataNorm[1:762, 1:4]
zero_y = dataNorm[1:762, 5]

hum_X = dataNorm[763:1372, 1:4]
hum_y = dataNorm[763:1372, 5]

H <- diag(c(0.75, 0.75, 0.75, 0.75))

zeroModel = kde(x=zero_X[,1:4], H = H)
humModel = kde(x=hum_X[,1:4], H = H)

#1, 100, 763, 1000
predict(zeroModel, x=dataNorm[1000,1:4])
predict(humModel, x=dataNorm[1000,1:4])


