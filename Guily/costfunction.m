function [loss] = costfunction( Ytrain, costChoice, W, Xtrain, B, activation)
global ptrain;
tmp = (W'* Xtrain);
repmat (B,1, size(Xtrain) );
ypred = tmp + B; 
pred = activationFunction(ypred, activation);
    
if isequal(costChoice,'MSE')
    D = abs(Ytrain-pred).^2;
    MSE = sum(D(:))/numel(Ytrain);
    loss = (MSE/2); 
elseif isequal(costChoice,'other')
    D = abs(Ytrain-pred).^2;
    MSE = sum(D(:))/numel(Ytrain);
    loss = (MSE/2);
end
  
end
