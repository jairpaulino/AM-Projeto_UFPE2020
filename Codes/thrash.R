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
              H = H, 
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
