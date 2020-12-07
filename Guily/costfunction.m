function [cost] = costfunction(pred , true)
  rep = true - pred;
  cost = 0 ;
  for i = 1:20
    
    cost += rep(i)**2;
    
  end
  
  cost = cost/40;
  
endfunction
