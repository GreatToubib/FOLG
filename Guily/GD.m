function [W, B, loss_history] = GD(W,B, Xts, y_encoded, lr, epoch_number, activation, abs_tol, rel_tol, patience)
patience_count=0;
loss_history=zeros(epoch_number);
for epoch = 1:epoch_number
    epoch
   
    ypred = W' * Xts + B; % ypred avant activation du test sample
    [a_ypred, ader] = activationFunction(ypred, activation);
    
    loss = costfunction(a_ypred, y_encoded', 'MSE'); % test de la cost function sur le test sample
    loss_history(epoch) = loss;
    
    pred_error = a_ypred-y_encoded';
    part3=  (pred_error .*  ader);
    gd = Xts * ( 1/20 * part3)'; 
    W = W - lr*gd;
    
    if epoch == 1 
        previous_loss = 10000;
    end

    diff = previous_loss - loss
    if diff  < abs_tol
        patience_count = patience_count+1;
        if patience_count==5
            disp('breaking abs')
            break
        end
    elseif diff < previous_loss* rel_tol
        patience_count = patience_count+1;
        if patience_count==5
            disp('breaking rel')
            break
        end
    else
        previous_loss = loss;
      
    end

end

end


	