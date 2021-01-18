function [loss] = costfunction( Ytrain, costChoice, W, Xtrain, B, activation)
% Fonction de cout, calcule la loss entre la prediction et la ground_truth
% de base, on utilise MSE, pas d'autre implémentée pour le moment

tmp = (W'* Xtrain);

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
