function [ W,B ] = gdFunction( Xbatch, Ybatch, activation, W, B , lr, batch_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    global ptrain;
    bicMac = repmat(B, 1, batch_size);
    ypred = (W'* Xbatch) + bicMac; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdW = Xbatch * ( 2/ptrain * part3)'; 
    W = W - lr*gdW; % update W
    % optimisation de B, a priori ca ameliore pas les resultats, l accuracy
    % sur le test est m�me moins bonne, de 0.64 decend a 0.62, et c est
    % entre 1.5 et 2 fois plus long. 
    ypred = (W'* Xbatch) + bicMac;
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdB = 2 * ( 1/ptrain * part3)'; 
    %gdB=mean(gdB,1);
    bicMac = bicMac - lr*gdB'; % update B
    B=bicMac(:,1);
end

