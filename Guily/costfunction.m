function [cost] = costfunction(pred , true)
  rep = true - pred;
  cost = 0 ;
  for i = 1:20
    squared = rep(i)^2;
    cost = cost + squared ;
    
  end
  
  cost = cost/40; % peka par 40 deja? parce que facteur 1/2 pour la derivation et 20 neurones. 
  % je pense pas que c est mnt qu il faut diviser ainsi, la loss function
  % se fait sur une epoh il me semble 
  
end
