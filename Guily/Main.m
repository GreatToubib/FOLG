%wow le projet 
clc;
clear all;
load('data_doc1.mat')

test_sample=1
[m,p] = size(Xts);
W = ones (20,m) * 0.001 ;
ypred = W * Xts (:,test_sample) ; % ypred avant activation du test sample
%ypred = W * Xts ; % ypred sur tout


% faudrait pas appliquer sigmoid a chaque output dans une boucle? ou en
% matriciel du coup? en matriciel pour stochastic, si on les prend tous d
% un coup.
%a = softmax(a) 
a_ypred = sigmoid(ypred);  % ypred avec activation sigmoid
%a = ReLU (a)
%sd = softmaxder(a) 
sd = sigmoidder (ypred);
%sr = ReLUder (a)


y_encoded = (yts==1:20); % one hot encoding des labels
ytrue = y_encoded (test_sample,:) % visualisation du label du test sample

cost = costfunction(a_ypred, ytrue) % test de la cost function sur le test sample

W_trained = GD(W, Xts, y_encoded,  0.01,50, 'sigmoid', 0.001 );
%for i = 1:20 
% a = W * Xts (:,i) ;
%  a = sigmoid(a) ;
%  sd = sigmoidder (a) ;
%  cost = costfunction (a, true)
  
%end
