%wow le projet 
clear all
clear clc
load('data.mat');
[m,p] = size(Xts);
W = ones (20,m) * 0.01 ; 

a = W * Xts (:,1) ;
a=a';
sa = softmaxOriginal(a) ;
sa

y_encoded = (yts==1:20);
wtf=y_encoded(1,:);
cost = costfunction(sa,wtf );

