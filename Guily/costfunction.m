function [loss] = costfunction(pred , true, costChoice)
    
if isequal(costChoice,'MSE')
    D = abs(true-pred).^2;
    MSE = sum(D(:))/numel(true);
    loss = (MSE/2); 
elseif isequal(costChoice,'other')
    D = abs(true-pred).^2;
    MSE = sum(D(:))/numel(true);
    loss = (MSE/2);
end
  
end
