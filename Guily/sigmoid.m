function [zs] = sigmoid(z)
  
  zs = zeros (1,20);

  for i = 1:size(z)
   
    zs(i) = 1/(1+e**(-z(i)));
  
  end
  
 zs = zs';
  
end