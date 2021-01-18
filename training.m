function [best_W, best_B, train_loss_history, val_loss_history] = training(W, B, Xtrain, Ytrain, Xtest, Ytest, epoch_number, activation, abs_tol, rel_tol, patience, batch_size)
patience_count=0;

global loss_batch_history;
global ptrain;
global k;
batch_number = floor(ptrain/batch_size)
lrW = 1;
lrB = 10^4;
k = 0;
W_moins_1 = W;
for epoch = 1: epoch_number
    epoch
    
    % shuffle training set at each epoch[m,p] = size(Xts);
    % Get a new order for the rows.
    newRowOrder = randperm(ptrain);
    % Apply that order to both arrays.
    Xtrain = Xtrain';
    Xtrain = Xtrain(newRowOrder, :);
    Xtrain = Xtrain';
    Ytrain = Ytrain(newRowOrder, :);
    for batch = 1: batch_number
        k=k+1;
        Xbatch = Xtrain(:,(batch-1)*batch_size+1:batch*batch_size); 
        Ybatch = Ytrain((batch-1)*batch_size+1:batch*batch_size,:);
        %gradient descent 
        [ W, B, lrW, lrB, W_moins_1] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB, W_moins_1);
       
    end 
    
    % training loss 
    loss = costfunction(Ytrain', 'MSE', W, Xtrain, B, activation); 
    train_loss_history(epoch) = loss;
    % validation loss 
    loss = costfunction(Ytest', 'MSE', W, Xtest, B, activation)
    val_loss_history(epoch) = loss;

    if epoch == 1 
        previous_loss = 10000;
    end
    
    diff = previous_loss - loss;
    if diff  < abs_tol
        patience_count = patience_count+1
        if patience_count==patience
            disp('breaking abs')
            break
        end
        
    else
        patience_count = 0;
    end
    %%%%%%%%%%%%%%%%% patience pour early stopping avec tolerance relative,
    %%%%%%%%%%%%%%%% a priori inutile. %%%%%%%%%%%%%%%%%%%
    %if diff < previous_loss* rel_tol
        %patience_count = patience_count+1
        %if patience_count==5
            %disp('breaking rel')
            %break
        %end
    %else
       % patience_count = 0;
    %end
    if patience_count == 0 
        previous_loss = loss;
        best_W = W;
        best_B = B;
    end

end

f1 = figure;
figure(f1);
plot(train_loss_history, 'b');
hold on;
plot(val_loss_history, 'r');
hold off;
f2 = figure;
figure(f2);
plot(loss_batch_history, 'r');
hold off;

end


	