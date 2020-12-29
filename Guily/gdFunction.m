function [ W, B, lrW, lrB] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    
    lrW = 1.5* lrW;
    global ptrain;
    bicMac = repmat(B, 1, batch_size);
    ypred = (W'* Xbatch) + bicMac; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdW = Xbatch * 1/20 * ( 1/batch_size * part3)'; 
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
    lrB = 1.5* lrB;
    ypred = (W'* Xbatch) + bicMac;
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdB = 1/20 * ( 1/batch_size * part3)';
    previous_bicMac = bicMac;
    previous_B= previous_bicMac(:,1);
    bicMac = bicMac - lrB*gdB'; % update B
    B=bicMac(:,1);
    %previous_B(1,1)
    %B(1,1)
    new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    old_loss = costfunction( Ytrain', 'MSE', W, Xtrain, previous_B, activation);
    while new_loss > old_loss
    lrB = lrB/2
    bicMac = previous_bicMac - lrB*gdB'; % update B
    B=bicMac(:,1);
    new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    end 
    
    
end

