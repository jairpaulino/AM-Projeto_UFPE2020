#density()

#a = density(dataNormTrain[,-5], 
#            bw = 1.5, 
            #adjust = 1,
#            kernel = "gaussian",
#       )
#plot(a)
#gausspr

# Compute kde for a diagonal bandwidth matrix (trivially positive definite)
H = diag(c(0.75, 0.75, 0.75, 0.75))
kde = ks::kde(x = dataNormTrain[,-5], 
              #H = H, 
              gridtype = "gaussian"
              )


zero_X = data[1:762, 1:4]
zero_y = data[1:762, 5]

hum_X = data[763:1372, 1:4]
hum_y = data[763:1372, 5]


# The eval.points slot contains the grids on x and y
#str(kde$eval.points)
#plot(kde$eval.points)

modelP_0 = mkde(as.matrix(zero_X),
              h = 1.75,
              thumb = 'none')

modelP_1 = mkde(as.matrix(hum_X),
                h = 1.75,
                thumb = 'none')

modelP_0[1]
modelP_1[1]

mkde.tune(as.matrix(iris[, 1:4]), c(0.1, 3) )
# 1) The optimal bandwidth value
# 2) The value of the pseudo-log-likelihood at that 
# given bandwidth value.
mkde.tune(as.matrix(zero_X), 
          low = 0.1, up = 3)

mkde.tune(as.matrix(hum_X), 
          low = 0.1, up = 3)



#View(modelP_0); View(modelP_1)
library("MASS")
data("birthwt")
attach(birthwt)
library(np)

data("Italy")
attach(Italy)
# First, compute the bandwidths... note that this may take a minute or
# two depending on the speed of your computer.
bw <- npcdensbw(formula=gdp~ordered(year))
# Next, compute the condensity object...
fhat <- npcdens(bws=bw)
# The object fhat now contains results such as the estimated conditional
# density function (fhat$condens) and so on...
summary(fhat)
# Call the plot() function to visualize the results (<ctrl>-C will
# interrupt on *NIX systems, <esc> will interrupt on MS Windows
# systems).
plot(bw)
detach(Italy)
# EXAMPLE 1 (INTERFACE=DATA FRAME): For this example, we load Giovanni
# Baiocchi's Italian GDP panel (see Italy for details), and compute the
# likelihood cross-validated bandwidths (default) using a second-order
# Gaussian kernel (default). Note - this may take a minute or two
# depending on the speed of your computer.
data("Italy")
attach(Italy)
# First, compute the bandwidths... note that this may take a minute or
# two depending on the speed of your computer.
# Note - we cast `X' and `y' as data frames so that plot() can
# automatically grab names (this looks like overkill, but in
# multivariate settings you would do this anyway, so may as well get in
# the habit).
X <- data.frame(year=ordered(year))
y <- data.frame(gdp)
bw <- npcdensbw(xdat=X, ydat=y)
# Next, compute the condensity object...
fhat <- npcdens(bws=bw)
# The object fhat now contains results such as the estimated conditional
# density function (fhat$condens) and so on...
summary(fhat)
# Call the plot() function to visualize the results (<ctrl>-C will
# interrupt on *NIX systems, <esc> will interrupt on MS Windows systems).
plot(bw)
detach(Italy)
# EXAMPLE 2 (INTERFACE=FORMULA): For this example, we load the old
# faithful geyser data from the R `datasets' library and compute the
# conditional density function.
library("datasets")
data("faithful")
attach(faithful)
# Note - this may take a few minutes depending on the speed of your
# computer...
bw <- npcdensbw(formula=eruptions~waiting)
summary(bw)
# Plot the density function (<ctrl>-C will interrupt on *NIX systems,
# <esc> will interrupt on MS Windows systems).
plot(bw)
detach(faithful)
# EXAMPLE 2 (INTERFACE=DATA FRAME): For this example, we load the old
# faithful geyser data from the R `datasets' library and compute the
# conditional density function.
library("datasets")
data("faithful")
attach(faithful)
# Note - this may take a few minutes depending on the speed of your
# computer...
# Note - we cast `X' and `y' as data frames so that plot() can


#######
library(KernSmooth)
attach(faithful)
fhat <- bkde(x=waiting)
plot (fhat, xlab="x", ylab="Density function")

data(geyser, package="MASS")
x <- geyser$duration
est <- bkde(x, bandwidth=0.25)
plot(est, type="l")

######
# Sample 100 points from a N(0, 1)
set.seed(1234567)
samp <- rnorm(n = 100, mean = 0, sd = 1)

# Quickly compute a kde and plot the density object
# Automatically chooses bandwidth and uses normal kernel
plot(density(x = samp))

# Select a particular bandwidth (0.5) and kernel (Epanechnikov)
lines(density(x = samp, bw = 0.5, kernel = "epanechnikov"), col = 2)
plot(density(x = samp, from = -4, to = 4), xlim = c(-5, 5))

# The density object is a list
kde <- density(x = samp, from = -5, to = 5, n = 1024)
str(kde)
## List of 7

# Note that the evaluation grid "x" is not directly controlled, only through
# "from, "to", and "n" (better use powers of 2)
plot(kde$x, kde$y, type = "l")
curve(dnorm(x), col = 2, add = TRUE) # True density
rug(samp)

K_Epa <- function(z, h = 1) 3 / (4 * h) * (1 - (z / h)^2) * (abs(z) < h)
mu2_K_Epa <- integrate(function(z) z^2 * K_Epa(z), lower = -1, upper = 1)$value

# Epanechnikov kernel by R
h <- 0.5
plot(density(0, kernel = "epanechnikov", bw = h))

# Build the equivalent bandwidth
h_tilde <- h / sqrt(mu2_K_Epa)
curve(K_Epa(x, h = h_tilde), add = TRUE, col = 2)

h_tilde <- 2
h <- h_tilde * sqrt(mu2_K_Epa)
curve(K_Epa(x, h = h_tilde), from = -3, to = 3, col = 2)
lines(density(0, kernel = "epanechnikov", bw = h))

zero_X = dataNorm[1:762, 1:4]
zero_y = dataNorm[1:762, 5]

hum_X = dataNorm[763:1372, 1:4]
hum_y = dataNorm[763:1372, 5]

H <- diag(c(0.5, 0.5, 0.5, 0.5))
kde_zero <- ks::kde(x = zero_X, H = H, binned = F)
kde_hum <- ks::kde(x = hum_X, H = H, binned = F)

kde_zero$cont
hist(kde_zero$cont)

vec_test = c(0.5, 0.3, 0.5, 0.6)

predict(kde_zero)

n <- 200
set.seed(35233)
x <- mvtnorm::rmvnorm(n = n, mean = c(0, 0),
                      sigma = rbind(c(1.5, 0.25), c(0.25, 0.5)))

# Compute kde for a diagonal bandwidth matrix (trivially positive definite)
H <- diag(c(1.25, 0.75))
kde <- ks::kde(x = x, H = H, binned = F)

# The eval.points slot contains the grids on x and y
str(kde$eval.points)
## List of 2
##  $ : num [1:151] -8.58 -8.47 -8.37 -8.26 -8.15 ...
##  $ : num [1:151] -5.1 -5.03 -4.96 -4.89 -4.82 ...

# The grids in kde$eval.points are crossed in order to compute a grid matrix
# where to evaluate the estimate
dim(kde$estimate)
## [1] 151 151

# Manual plotting using the kde object structure
image(kde$eval.points[[1]], kde$eval.points[[2]], kde$estimate,
      col = viridis::viridis(20), xlab = "x", ylab = "y")
points(kde$x) # The data is returned in $x


# 
dt = iris
H <- diag(c(0.5, 0.5, 0.5, 0.5))
kde_0 <- ks::kde(x = dt[1:50, 1:4], H = H, binned = F)
kde_1 <- ks::kde(x = dt[51:100, 1:4], H = H, binned = F)
kde_2 <- ks::kde(x = dt[101:150, 1:4], H = H, binned = F)

kde_0$cont[1]
kde_1$cont[1]
kde_2$cont[1]

library(MASS)
x1 <- crabs[crabs$sp=="B", 4]
x2 <- crabs[crabs$sp=="O", 4]
loct <- kde.local.test(x1=x1, x2=x2)
plot(loct)

set.seed(8192)
samp <- 1000
x <- rnorm.mixt(n=samp, mus=0, sigmas=1, props=1)
y <- rnorm.mixt(n=samp, mus=0, sigmas=1, props=1)
kde.test(x1=x, x2=y)$pvalue ## accept H0: f1=f2
library(MASS)
data(crabs)
x1 <- crabs[crabs$sp=="B", c(4,6)]
x2 <- crabs[crabs$sp=="O", c(4,6)]
kde.test(x1=x1, x2=x2)$pvalue ## reject H0: f1=f2


