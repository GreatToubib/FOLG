function [MSE] = costfunction(pred , true)

    D = abs(true-pred).^2;
    MSE = sum(D(:))/numel(true);
  
    MSE = (MSE/2); 
  
  
end
