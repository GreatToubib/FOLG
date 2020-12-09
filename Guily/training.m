function [W, B, train_loss_history, val_loss_history] = training(W, B, Xtrain, Ytrain, Xtest, Ytest, lr, epoch_number, activation, abs_tol, rel_tol, patience, batch_size)
patience_count=0;

global ptrain;
batch_number= floor(ptrain/batch_size)

for epoch = 1: epoch_number
    epoch
    
    for batch = 1: batch_number
        Xbatch = Xtrain(:,(batch-1)*batch_size+1:batch*batch_size); 
        Ybatch = Ytrain((batch-1)*batch_size+1:batch*batch_size,:);
        %gradient descent 
        [ W ] = gdFunction(Xbatch ,Ybatch,  activation, W, B , lr, batch_size);
    end 
    
    % training loss 
    tmp = (W'* Xtrain);
    %size(tmp)
    %size(B)
    ypred = tmp + B; 
    a_ypred = activationFunction(ypred, activation);
    loss = costfunction(a_ypred, Ytrain', 'MSE'); 
    train_loss_history(epoch) = loss;
    % validation loss 
    tmp = (W'* Xtest);
    %size(tmp)
    %size(B)
    ypred = tmp + B; 
    a_ypred = activationFunction(ypred, activation);
    loss = costfunction(a_ypred, Ytest', 'MSE')
    val_loss_history(epoch) = loss;

    if epoch == 1 
        previous_loss = 10000;
    end

    diff = previous_loss - loss;
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


	