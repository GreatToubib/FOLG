function [ W, B, lrW, lrB, W_moins_1 ] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB, W_moins_1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optimisation de W
    % optimisation de W
    global k;
    global loss_batch_history;
    global accel;
    global accelChoice;
    
    lrW = 1.5* lrW;
    bicMac = repmat(B, 1, batch_size);
    % implementer nesterov ici , on rajoute le momentum dans W
    ypred = (W'* Xbatch) + bicMac; 
    [a_ypred, ader] = activationFunction(ypred, activation);
    pred_error = a_ypred-Ybatch';
    part3=  (pred_error .*  ader);
    gdW = Xbatch * 1/20 * ( 1/batch_size * part3)'; 
    W_moins_2 = W_moins_1;
    W_moins_1 = W;
    W = updateW( W_moins_1, lrW,gdW , W_moins_2);
    new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    old_loss = costfunction( Ytrain', 'MSE', W_moins_1, Xtrain, B, activation);
    % ok so the problem with heavyball is that the contribution of the momemtum becomes too important and fucks things up. 
    % no matter the lr, momemtum creates a worse new loss and we get stuck. there is no guarantee to go down
    % so the solution is to deactivate momemtum for one step. 
    while new_loss > old_loss 
        accel = 'normal';
        lrW = lrW/2;
        W = updateW( W_moins_1, lrW,gdW, W_moins_2 );
        accel= accelChoice;
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
        lrB = lrB/2;
        bicMac = previous_bicMac - lrB*gdB'; % update B
        B=bicMac(:,1);
        new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    end 
    
    loss_batch_history(k)=new_loss;
end

