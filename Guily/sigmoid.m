function [zs] = sigmoid(z)
  zs = zeros (13960,20);
  for j = 1:13960

      for i = 1:20
        zrow=z(j)
        zs(j,i) = 1/(1+exp(-zrow(i)));

      end

     zs = zs';
  
end