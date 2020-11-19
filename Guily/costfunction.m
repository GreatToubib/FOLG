function [cost] = costfunction(pred , true)
  rep = pred - true;
  cost = 0 ;
  for i = 1:20
    
    cost += rep(i)**2;
    
  end
  
  cost = cost/20;
  
endfunction
