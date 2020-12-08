function [W, B, loss_history] = training(W, B, Xtrain, Ytrain, lr, epoch_number, activation, abs_tol, rel_tol, patience, batch_size)
patience_count=0;
loss_history=zeros(epoch_number);
global ptrain;
batch_number= floor(ptrain/batch_size)
for epoch = 1:epoch_number
    epoch
    
    for batch = 1:batch_number
        Xbatch = Xtrain(:,(batch-1)*batch_size+1:batch*batch_size); 
        Ybatch = Ytrain((batch-1)*batch_size+1:batch*batch_size,:);
        %gradient descent 
        [ W, B ] = gdFunction(Xbatch ,Ybatch,  activation, W, B , lr);
    end 
    
    % loss 
    ypred = W'* Xtrain + B; 
    a_ypred = activationFunction(ypred, activation);
    loss = costfunction(a_ypred, Ytrain', 'MSE'); 
    loss_history(epoch) = loss;

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


	