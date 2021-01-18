function [best_W, B, train_loss_history] = trainingAll(W, B, Xtrain, Ytrain, epoch_number, activation, abs_tol, rel_tol, patience, batch_size)
% entrainement du reseau de neurones sans validation, sur 100& donnees. 
% Appelle gdFunction sur chaque batch de chaque epoch puis mesure les
% performances a chaque epoch. l'entrainement s'arrete via un early stopper
% ( si la loss ne s'ameliore plus assez pendant un certain nombre
% (patience) d'epochs) ou apprès un nombre max d'epoch (epoch_number) 
%OUPUTS : best_W: meilleurs W enregistrés
%         best_B: meilleurs B enregistrés
%         train_loss_history: historique de la loss à l'entrainement 
%         val_loss_history historique de la loss à la validation 

patience_count=0;

global loss_batch_history;
global ptrain;
global k;
batch_number = floor(ptrain/batch_size)
lrW = 1;
lrB = 10^4;
k = 0;
W_moins_1 = W; % initialisation de W_moins_1 pour heavyball method

%%%%%%%%%%%%%%% iteration a travers chaque epoch 
for epoch = 1: epoch_number
    epoch % print epoch
    
    %%%%%%%%%%%%%%%%% shuffle training set at each epoch[m,p] = size(Xts);
    % Get a new order for the rows.
    newRowOrder = randperm(ptrain);
    % Apply that order to both arrays.
    Xtrain = Xtrain';
    Xtrain = Xtrain(newRowOrder, :);
    Xtrain = Xtrain';
    Ytrain = Ytrain(newRowOrder, :);
    
    
    %%%%%%%%%%%%%%% iteration a travers chaque batch 
    for batch = 1: batch_number
        k=k+1;
        Xbatch = Xtrain(:,(batch-1)*batch_size+1:batch*batch_size); 
        Ybatch = Ytrain((batch-1)*batch_size+1:batch*batch_size,:);
        % gradient descent sur le batch. 
        [ W, B, lrW, lrB, W_moins_1] = gdFunction( Xbatch, Ybatch, Xtrain, Ytrain, activation, W, B , batch_size, lrW, lrB, W_moins_1);
       
    end 
    
    % calcul training loss 
    loss = costfunction(Ytrain', 'MSE', W, Xtrain, B, activation)
    train_loss_history(epoch) = loss;

    %%%%%%%%%%% EARLY STOPPER 
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
    %%%%%%%%%%%%%%%% a priori inutile, early stopper avec val_abs suffit. 
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



end


	

	