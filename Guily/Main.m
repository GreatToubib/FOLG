%wow le projet 

%%% TODO
% optimiser le biais aussi. 
% adaptive learning rate 
% avec les conditions et tout l�. voir slides
%https://towardsdatascience.com/calculating-gradient-descent-manually-6d9bee09aa0b

clc;
clear all;
load('data_doc1.mat')
tic;
global mtrain;
global ptrain;
[m,p] = size(Xts);
batch_size= 512; % 11168 64 512 32
splitChoice='random';
splitValue=0.8;
initializationChoice = 'zeros'; % random, he, zeros, xavier or number
activation='sigmoid'; % 'relu' ou 'softmax' 'sigmoid'
abs_tol=10^-6;
rel_tol=10^-6;
epoch_number=20;
patience=5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = initializationFunction(m,initializationChoice); 
Winit=W; % store initial weights

% a priori random le mieux
% https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
%When we have a sigmoid activation function, it is better to use Xavier Glorot initialization of weights.
% When we have ReLU activation function, it is better to use He-initialization of weights.
B= zeros(20,1); % zeros c est tres bien, l initialisation des biais change rien askip
y_encoded = (yts==1:20); % one hot encoding des labels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% splitting data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xtrain,Ytrain, Xtest, Ytest] = createTrainingSet(Xts,yts, splitChoice, splitValue);
[mtrain,ptrain] = size(Xtrain);
ytrain_encoded = (Ytrain==1:20);
ytest_encoded = (Ytest==1:20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% back propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[W_trained, B_trained, train_loss_history, val_loss_history]= training(W, B, Xtrain, ytrain_encoded, Xtest, ytest_encoded, epoch_number, activation, abs_tol, rel_tol,patience, batch_size);
[W_trained, B_trained, train_loss_history]= training_on_all(W, B, Xtrain, ytrain_encoded, epoch_number, activation, abs_tol, rel_tol,patience, batch_size);

training_time = toc
%accuracy = validation( Xts, yts, W_trained );
%accuracy = validation( Xtrain,Ytrain, W_trained );
%accuracy = validation( Xtest, Ytest, W_trained, B_trained );

plot(train_loss_history, 'b');
%hold on;
%plot(val_loss_history, 'r');
hold off;

createFile( Xvr, W_trained, B_trained )

