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

# The eval.points slot contains the grids on x and y
str(kde$eval.points)
plot(kde$eval.points)

modelP = mkde(as.matrix(dataNormTrain[,-5]),
              h = 1.75,
              thumb = 'none')
