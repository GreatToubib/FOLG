%wow le projet 
clear all;
load('data_doc1.mat')
[m,p] = size(Xts);
W = ones (20,m) * 0.001 ; 
a = W * Xts (:,1) ;

%a = softmax(a) 
a = sigmoid(a) 
%a = ReLU (a)
%sd = softmaxder(a) 
sd = sigmoidder (a) 
%sr = ReLUder (a)


y_encoded = (yts==1:20);
true = y_encoded (1,:);

cost = costfunction (a, true)

alpha = 0.001;

for i = 1:20 
  a = W * Xts (:,i) ;
  a = sigmoid(a) ;
  sd = sigmoidder (a) ;
  cost = costfunction (a, true)
  
end