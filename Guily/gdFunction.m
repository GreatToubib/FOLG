function [ W, B, lrW, lrB] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    lrW = 1.5* lrW;
    %lrB = 1.5* lrB;
    global ptrain;
    bicMac = repmat(B, 1, batch_size);
    ypred = (W'* Xbatch) + bicMac; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdW = Xbatch * ( 2/ptrain * part3)'; 
    previous_W = W;
    W = previous_W - lrW*gdW; % update W
    new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    old_loss = costfunction( Ytrain', 'MSE', previous_W, Xtrain, B, activation);
    while new_loss > old_loss
    lrW = lrW/2;
    W = previous_W - lrW*gdW; % update W
    new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    end 
   
    
    
    % optimisation de B, a priori ca ameliore pas les resultats, l accuracy
    % sur le test est même moins bonne, de 0.64 decend a 0.62, et c est
    % entre 1.5 et 2 fois plus long. 
    ypred = (W'* Xbatch) + bicMac;
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdB = 2 * ( 1/ptrain * part3)'; 
    %gdB=mean(gdB,1);
    bicMac = bicMac - lrB*gdB'; % update B
    B=bicMac(:,1);
end

