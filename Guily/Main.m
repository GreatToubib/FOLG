%wow le projet 
[m,p] = size(Xts);
W = ones (20,m) * 0.01 ; 

a = W * Xts (:,1) ;

a = softmax(a) ;

y_encoded = (yts==1:20);
true = y_encoded (1);

cost = costfunction (a, true);

