function [zd] = sigmoidder(z)
  
  zd = zeros (1,20);

  for j = 1:size(z)
   
    zd(j) = (1/(1+e**(-z(j)))) * (1- (1/(1+e**(-z(j)))) ) ;
  
  end
  
 zd = zd';
  
end