function [sa] = softmaxOriginal(z)
  
  zs = zeros (1,20);
  somme = 0;
  for i = 1:size(z,2)
   
      zs(i) = exp(z(i))
      somme = somme + exp(z(i))
  end
  sa = zs./somme;
  
  
end
