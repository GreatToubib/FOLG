%wow le projet 
clc;
%clear all;
load('data_doc1.mat')

test_sample=1;
[m,p] = size(Xts);
W = ones (m,20) * 0.01 ;
B= zeros(1,20);
ypred = W' * Xts (:,test_sample) ; % ypred avant activation du test sample
%ypred = W * Xts ; % ypred sur tout


% faudrait pas appliquer sigmoid a chaque output dans une boucle? ou en
% matriciel du coup? en matriciel pour stochastic, si on les prend tous d
% un coup.
%a = softmax(a) 
a_ypred = sigmoidFunction(ypred);  % ypred avec activation sigmoid
%a = ReLU (a)
%sd = softmaxder(a) 
%sd = sigmoidder (ypred);
ader = sigmoidFunction(ypred) .* ( 1 - sigmoidFunction(ypred));
%sr = ReLUder (a)


y_encoded = (yts==1:20); % one hot encoding des labels
ytrue = y_encoded (test_sample,:) % visualisation du label du test sample
cost = costfunction(a_ypred, ytrue) % test de la cost function sur le test sample


lr=0.01;
activation='sigmoid';
abs_tol=10^-5
rel_tol=10^-5;% doit s ameliorer de 1 % ou on arrete
epoch_number=200;
%W_trained = GD(W, Xts, y_encoded, lr, epoch_number, activation, abs_tol, rel_tol);





% validation 
count=0;
for test_sample = 1:p
    [m,p] = size(Xts);
    ypred = W_trained' * Xts (:,test_sample) ; 
    a_ypred = sigmoidFunction(ypred);  % ypred avec activation sigmoid
    [M,I] = max(a_ypred);
    ytrue = yts(test_sample); % visualisation du label du test sample
    if ytrue== I 
        count= count+1;
    end
end
accuracy=count/p
