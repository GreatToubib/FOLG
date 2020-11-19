function [laperte] = costfunction(pred , true)
  rep = pred - true;
  laperte = 0 ;
  for i = 1:20
    laperte = laperte + rep(i)^2;
  end
  
  laperte = laperte/20;
  
end
