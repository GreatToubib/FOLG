%wow le projet 

%%% TODO
% Wolfe
% experiences
%https://towardsdatascience.com/calculating-gradient-descent-manually-6d9bee09aa0b

clc;
%clear all;
load('data_doc1.mat')

global loss_batch_history;
loss_batch_history=[];
global mtrain;
global ptrain;
global accelChoice;
global accel;
global k;
global only_W;

[m,p] = size(Xts);
batch_size= 256; % 11307 64 512 32 128 256 1
splitChoice='random';  % all(cest random aussi) , random 
splitValue=0.81;
accelChoice='normal'; %  normal , heavyball,                  nesterov pas implémenté
accel=accelChoice;
only_W = 0; % 1, onlyW,  0, on train W et B 
initializationChoice = 'zeros'; % random, he, zeros, xavier or int : tous les W sont initialisés à initializationChoice
activation='sigmoid'; % 'relu' ou 'softmax' 'sigmoid'
abs_tol=10^-6; % tol absolue sur la validation d'une epoch a l'autre pour l early stopper
rel_tol=10^-6; % tol relative sur la validation d'une epoch a l'autre pour l early stopper
epoch_number=100; % nbre d epoch max si earlys topper s enclenche pas
epoch_number_on_all=10; % epoch pour train sur tout, ou on a pas de valdiation set et donc pas d'early stopper
patience=5; % nombre d epoch de patience pour early stopper 
global lrW;
lrW=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% splitting data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% on veut entrainer sur le meme training set a chaque fois.
[Xtrain,Ytrain, Xtest, Ytest] = createTrainingSet(Xts,yts, splitChoice, splitValue);
[mtrain,ptrain] = size(Xtrain);
ytrain_encoded = (Ytrain==1:20);
if not(isequal(splitChoice,'all'))
    ytest_encoded = (Ytest==1:20);
end

        
performances = []
perf_i=0
for batch_size = [ 512 1024 ] %128 512 1024
    init_int=0
    if batch_size == 11307
        patience=10;
    else
        patience=5;
    end
    for accelChoice = {'normal' , 'heavyball'} %
        
        accelChoice = accelChoice{1} %  normal , heavyball,                  nesterov pas implémenté
        accel=accelChoice;
        tic;
        perf_i=perf_i+1;
        batch_size
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W = initializationFunction(m,initializationChoice); 
        Winit = W; % store initial weights

        % a priori random le mieux
        % https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
        %When we have a sigmoid activation function, it is better to use Xavier Glorot initialization of weights.
        % When we have ReLU activation function, it is better to use He-initialization of weights.
        B= zeros(20,1); % zeros c est tres bien, l initialisation des biais change rien askip
        y_encoded = (yts==1:20); % one hot encoding des labels


       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if isequal(splitChoice,'all')
            [W_trained, B_trained, train_loss_history]= trainingAll(W, B, Xtrain, ytrain_encoded, epoch_number_on_all, activation, abs_tol, rel_tol,patience, batch_size);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% accuracy sur training test %%%%%%%%%%%
            accuracy = validation( Xtrain,Ytrain, W_trained, B_trained );
        else
            [W_trained, B_trained, train_loss_history, val_loss_history,epoch]= training(W, B, Xtrain, ytrain_encoded, Xtest, ytest_encoded, epoch_number, activation, abs_tol, rel_tol,patience, batch_size);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Validation %%%%%%%%%%%
            %accuracy = validation( Xts, yts, W_trained );
            accuracy = validation( Xtest, Ytest, W_trained, B_trained ) *100;
        end
        training_time = toc
        
        performances(perf_i,:) = [batch_size, training_time, accuracy, epoch] ;
        %%%%%%%%%%%%%%plot des courbes de trainign et validation %%%%%%%%%%%%%
        f1 = figure;
        figure(f1);
        plot(train_loss_history, 'b');
        hold on;
        plot(val_loss_history, 'r');
        title('loss par epoch')
        legend('train','val')
        hold off;
        %f2 = figure;
        %figure(f2);
        %plot(loss_batch_history, 'b');
        %title('train loss par batch')
        %hold off;
        filename1=strcat('results/f1_',int2str(batch_size),'_',accelChoice,'.jpg');
        %filename2=strcat('results/f2_',int2str(batch_size),'_',initializationChoice,'.jpg');
        saveas(f1,filename1)
        %saveas(f2,filename2)
    end
end
performances
save('results/performances_batch_accelChoice.mat','performances')




