library("biOps")
library("FactoMineR")
library("ade4")

f<-function(n,x){}
x <- readJpeg("./Data/google/image-3.jpeg")
plot(x)
cl<-kmeans(x, 2)
#cluster
noire<-blanc<-z<-y<-x
levels(factor(cl$cluster))
for (i in 1:57){for (j in 1:300){for (k in 1:3){noire[i,j,k]<-0}}}
for (i in 1:57){for (j in 1:300){for (k in 1:3){blanc[i,j,k]<-255}}}
for (i in 1:57){for (j in 1:300){for (k in 1:3){y[i,j,k]<-(cl$cluster[i + (j-1)*57 + (k-1)*57*300]-1)*x[i,j,k]}}}
for (i in 1:57){for (j in 1:300){for (k in 1:3){z[i,j,k]<-(-cl$cluster[i + (j-1)*57 + (k-1)*57*300]+2)*x[i,j,k]}}}
levels(factor(y))
plot(noire)
legend(0,-1,"noire")
plot(blanc)
legend(0,-1, "blanc")
plot(y)
plot(z)



