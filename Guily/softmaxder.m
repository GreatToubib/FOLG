function [z] = softmaxder(zs)

zd = zeros (1,20);
 
for i = 1:20
  for j = 1:20
    if (i == j)
     
      zd (i,j) = zs(i)* (1-zs(i));
    else
      zd (i,j) = (-zs(i))* zs(j);
    
  end
  
  end
end
  #z = zd (:,1);
  z = zd';
end