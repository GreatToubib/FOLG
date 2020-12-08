function [ W,B ] = gdFunction( Xtrain, Ytrain, activation, W, B , lr)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    global ptrain;
    ypred = W'* Xtrain + B; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ytrain';
    part3=  (pred_error .*  ader);
    gdW = Xtrain * ( 2/ptrain * part3)'; 
    W = W - lr*gdW; % update W
    
    % optimisation de B, a priori ca ameliore pas les resultats, l accuracy
    % sur le test est même moins bonne, de 0.64 decend a 0.62, et c est
    % entre 1.5 et 2 fois plus long. 
    ypred = W'* Xtrain + B; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ytrain';
    part3=  (pred_error .*  ader);
    gdB = 2 * ( 1/ptrain * part3)'; 
    B = B - lr*gdB'; % update B
end

