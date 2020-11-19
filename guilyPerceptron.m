%wow le projet 
[m,p] = size(Xts);
W = ones (20,m); 

a = W * Xts ( :,1) ;

a = softmax(a) 