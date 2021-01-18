function [zr] = ReLUder(z)
  
  zr = zeros (1,20);

  for i = 1:size(z) 
    if (z(i)>0) 
     
      zr(i) = 1 ;
    
    end  
  
  end
  
 zr = zr';
  
end