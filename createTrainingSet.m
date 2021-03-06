function [Xtrain,Ytrain, Xtest, Ytest] = createTrainingSet(Xts,y_encoded, splitChoice, splitValue)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    Xts = Xts';
    [m,p] = size(Xts);
    if isequal(splitChoice,'random') % on melange tout puis on split en proportion splitValue
        numberOfRows = m;
        % Get a new order for the rows.
        if exist('newRowOrder.mat','file') == 2
             newRowOrder = load('newRowOrder.mat')
             newRowOrder=newRowOrder.newRowOrder;
        else
             newRowOrder = randperm(numberOfRows);
             save('newRowOrder.mat','newRowOrder')
        end
        % Apply that order to both arrays.
        new_Xts = Xts(newRowOrder, :);
        new_Y_encoded = y_encoded(newRowOrder, :);
        Xtrain = new_Xts( 1:splitValue*numberOfRows, : ) ;
        Xtest = new_Xts( splitValue*numberOfRows:end , : ) ;
        Ytrain = new_Y_encoded( 1:splitValue*numberOfRows, : ) ;
        Ytest = new_Y_encoded( splitValue*numberOfRows:end , : ) ;
    end
    
    if isequal(splitChoice,'all') % tout en training, rien en test
       numberOfRows = m;
       % Get a new order for the rows.
       
       newRowOrder = randperm(numberOfRows);
             
       % Apply that order to both arrays.
       new_Xts = Xts(newRowOrder, :);
       new_Y_encoded = y_encoded(newRowOrder, :);
       Xtrain = new_Xts ;
       Ytrain = new_Y_encoded;
       Xtest=[];
       Ytest=[];
    end
    
    Xtrain=Xtrain';
    Xtest=Xtest';
   

end

