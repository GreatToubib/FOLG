function [zs] = softmax(z)
  
  zs = zeros (1,20);
  somme = 0;
  for i = 1:size(z)
   
  zs(i) = e**z(i);
  somme += e**z(i)  ;
  end
  
  
  
  zs = zs./somme ; 
  zs = zs';
  
end
