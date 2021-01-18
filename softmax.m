function [zs] = softmax(z)
  
  zs = zeros (1,20);
  somme = 0;
  for i = 1:size(z)
   
  zs(i) = exp(z(i));
  somme = somme + exp(z(i))  ;
  end
  
  
  
  zs = zs./somme ; 
  zs = zs';
  
end
