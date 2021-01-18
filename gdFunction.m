function [ W, B, lrW, lrB, W_moins_1 ] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB, W_moins_1)
%gdFunction Summary
% gradient descent sur le batch. 
    % si batch_size=all, c'est une GD classique,
    % si    1 < batch_size < all   , ca fait une minibatchGD
    % si = 1, ca donne une SGD mais apparemment tres peu performante.
    % surement parce qu'a chaque batch on calcule plein de fois la
    % loss et la du coup ca devient trop lourd. 
    
  %OUTPUTS:
    % W, W actualisés
    % B, B actualisés
    % lrW, learning rate sur W actualisé
    % lrB, actualisé
    % W_moins_1, les W précédents ( ceux  donnés en input du coup), utiles
    % pour methode accélérées (heavyball )
    
    
% optimisation de W
    % optimisation de W
    global k;
    global loss_batch_history;
    global accel;
    global accelChoice;
    global only_W;

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
    % ok so one problem with heavyball is that the contribution of the momemtum becomes too important and fucks things up. 
    % no matter the lr, momemtum creates a worse new loss and we get stuck. there is no guarantee to go down
    % so the solution is to deactivate momemtum for one step. 
    while new_loss > old_loss 
        accel = 'normal';  % desactivation de l'acceleratio  pour une update, ainsi on est sur de s'ameliorer 
        lrW = lrW/2;
        W = updateW( W_moins_1, lrW,gdW, W_moins_2 );
        accel= accelChoice; % on reactive l'acceleration
        new_loss = costfunction( Ytrain', 'MSE', W, Xtrain, B, activation);
    end
   
    % optimisation de B, a priori ca ameliore pas les resultats
    if only_W == 0 % choix de train les B ou non, voir Main
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
    end
    loss_batch_history(k)=new_loss;
end

