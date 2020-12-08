%wow le projet 

%%% TODO
% optimiser le biais aussi. 
% adaptive learning rate 
% avec les conditions et tout l�. voir slides
%

clc;
%clear all;
load('data_doc1.mat')
tic;
[m,p] = size(Xts);
splitChoice='random';
splitValue=0.8;
initializationChoice = 'zeros'; % random, he, zeros, xavier or number
activation='sigmoid'; % 'relu' ou 'softmax' 'sigmoid'
lr=10^-2;
abs_tol=10^-5;
rel_tol=10^-5;
epoch_number=50;
patience=5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = initializationFunction(m,initializationChoice); 
Winit=W; % store initial weights
% a priori random le mieux
% https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
%When we have a sigmoid activation function, it is better to use Xavier Glorot initialization of weights.
% When we have ReLU activation function, it is better to use He-initialization of weights.
B= zeros(20,p); % zeros c est tres bien, l initialisation des biais change rien askip
y_encoded = (yts==1:20); % one hot encoding des labels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% splitting data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xtrain,Ytrain, Xtest, Ytest] = createTrainingSet(Xts,yts, splitChoice, splitValue);
ytrain_encoded = (Ytrain==1:20);
ytest_encoded = (Ytest==1:20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% back propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[W_trained, B_trained, loss_history]= GD(W, B, Xtrain, ytrain_encoded, lr, epoch_number, activation, abs_tol, rel_tol,patience);

training_time = toc
%accuracy = validation( Xts, yts, W_trained );
%accuracy = validation( Xtrain,Ytrain, W_trained );
accuracy = validation( Xtest, Ytest, W_trained );



