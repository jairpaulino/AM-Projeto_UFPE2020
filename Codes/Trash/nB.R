library(naivebayes)
## naivebayes 0.9.7 loaded
### Simulate data
n <- 100
set.seed(1)
data <- data.frame(class = sample(c("classA", "classB"), n, TRUE),
                   bern = sample(LETTERS[1:2], n, TRUE),
                   cat = sample(letters[1:3], n, TRUE),
                   logical = sample(c(TRUE,FALSE), n, TRUE),
                   norm = rnorm(n),
                   count = rpois(n, lambda = c(5,15)))
train <- data[1:95, ]
test <- data[96:100, -1]
### General usage via formula interface
nb <- naive_bayes(class ~ ., train, usepoisson = TRUE)
summary(nb)
library(naivebayes)
## naivebayes 0.9.7 loaded
### Simulate data
n <- 100
set.seed(1)
data <- data.frame(class = sample(c("classA", "classB"), n, TRUE),
                   bern = sample(LETTERS[1:2], n, TRUE),
                   cat = sample(letters[1:3], n, TRUE),
                   logical = sample(c(TRUE,FALSE), n, TRUE),
                   norm = rnorm(n),
                   count = rpois(n, lambda = c(5,15)))
train <- data[1:95, ]
test <- data[96:100, -1]

### General usage via formula interface
nb <- naive_bayes(class ~ ., train, usepoisson = TRUE)
summary(nb)

# Classification
#predict(nb, test, type = "class")
# Alternatively
nb %class% test
# Alternatively
nb %prob% test
tables(nb, 1)


?bw.SJ

