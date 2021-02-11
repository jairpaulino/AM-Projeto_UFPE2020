#rm(list=ls())

#d is dichotomous and created as 30% of 10.000 observations d = 1.
d = rep(0, times = 7000)
d = c(d, rep(1, times = 3000))

#m and s are mean and standard deviation for the random normal distribution. mi and ma are min
#and max for random uniform distribution.
m = 5; s = 3; mi = 0.1; ma = 5

#Creating highly correlated features x and y.
set.seed(123)
x = runif(1000, min = mi, max = ma) * 0.2 + rnorm(1000, mean = m, sd = s) +
    d * -10 * runif(1000, min = mi / 2, max = ma / 2)
hist(x, nclass = 100)


y = -0.1*x+rnorm(1000, mean = 1, sd = 1.16)+3*d
hist(y, nclass = 100)

#Correlation between x and y is about -78%
cor(x, y); cor(y, d)
plot(x, d)
plot(y, d)

###TEST DATA####
dt = rep(0, times = 7000)
dt = c(dt, rep(1, times = 3000))


#change the distribution parameters to take occurring trends into account
mt = 5.1; st = 2.9; mit = 0.2; mat = 4.9
set.seed(321)
xt = runif(1000, min = mit, max = mat) * 0.2 + rnorm(1000, mean = mt, sd = st) +
 
  dt * -10 * runif(1000, min = mit / 2, max = mat / 2)
hist(xt, nclass = 100)
yt = -0.1*xt+rnorm(1000, mean = 1, sd = 1.16)+3*dt
hist(yt, nclass = 100)
cor(xt, yt); cor(yt, dt)
plot(xt,d); plot(yt,d)

#Computing the prior probabilities. Actually, they does not have to be calculated for this example ;-)
a_priori = sum(d)/length(d)
a_PrioriNOT = 1 - sum(d)/length(d)

#Computing the KDE for x and y given d = 1 and d = 0. This ist the model training.
KDxd1 = density(x[d==1], bw = 1)
plot(KDxd1, main = "KDE p(x|d=1)")

KDxd0 = density(x[d==0], bw = 1)
plot(KDxd0, main = "KDE p(x|d=0)")

KDy1 = density(y[d==1], bw = 1)
plot(KDy1, main = "KDE p(y|d=1)")

KDy0 = density(y[d==0], bw = 1)
plot(KDy0, main = "KDE p(y|d=0)")

#Computing the likelihoods p(x|d=1), p(x|d=0), p(y|d=1), and p(y|d=0). The approx function
#calculates a linear approximation between given points of the KDE:
likelyhoodx1 = approx(x = KDxd1$x, KDxd1$y, xt, yleft = 0.00001, yright = 0.00001)
likelyhoodx0 = approx(x = KDxd0$x, KDxd0$y, xt, yleft = 0.00001, yright = 0.00001)
likelyhoody1 = approx(x = KDy1$x, KDy1$y, yt, yleft = 0.00001, yright = 0.00001)
likelyhoody0 = approx(x = KDy0$x, KDy0$y, yt, yleft = 0.00001, yright = 0.00001)

#Computing the evidence. Note: the likelihoodxxx$y is the density:
evidence = likelyhoody1$y*likelyhoodx1$y*a_priori+likelyhoody0$y*likelyhoodx0$y*a_PrioriNOT

#Computing the a posteriori probability by using Bayes theorem:
a_posteriori = likelyhoody1$y*likelyhoodx1$y*a_priori/evidence

#assigning the prediction with a cutoff of 50%.
prediction = ifelse(a_posteriori > 0.5, 1, 0)

#calculating content of confusion matrix
hit1 = ifelse(prediction == 1 & dt == 1, 1, 0)
false1 = ifelse(prediction == 1 & dt == 0, 1, 0)
hit0 = ifelse(prediction == 0 & dt == 0, 1, 0)
false0 = ifelse(prediction == 0 & dt == 1, 1, 0)

#Creating confusion matrix
Confusion <- matrix(c(sum(hit1), sum(false0), sum(false1), sum(hit0)), nrow = 2, ncol = 2)
colnames(Confusion) <- c("d = 1", "d = 0")
rownames(Confusion) <- c("p(d = 1|x,y)", "p(d = 0|x,y)")

#Looking at the areas where misclassifications occur. Red = misclassification; Black = correct classified:

color = ifelse(prediction == dt, 1, 2)
plot(xt, dt, col = color)
plot(yt, dt, col = color)


