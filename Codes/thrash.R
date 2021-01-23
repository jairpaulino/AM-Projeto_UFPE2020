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
