function [ W, B, lrW, lrB] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    
    
    
    
    global ptrain;
    
    bicMac = repmat(B, 1, batch_size);
    bigMac = repmat (B,1, ptrain );
    ypred = (W'* Xbatch) + bicMac; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdW = Xbatch * 1/20 * ( 1/batch_size * part3)'; 
    previous_W = W;
    
    lrW = wolfe (W, B, gdW, lrW, Ytrain, Xtrain);
    W = previous_W - lrW*gdW; % update W
   
   
   
    % optimisation de B, a priori ca ameliore pas les resultats, l accuracy
    
    ypred = (W'* Xbatch) + bicMac;
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdB = 1/20 * ( 1/batch_size * part3)';
    gdB = gdB (1,:);
    previous_bicMac = bicMac;
    previous_B= previous_bicMac(:,1);
    
    lrB = wolfeB (W, B, gdB, lrB, Ytrain, Xtrain);
    bicMac = bicMac - lrB*gdB'; % update B
    B=bicMac(:,1);
    
    
end

