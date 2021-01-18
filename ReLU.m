function [zr] = ReLU(z)
  
  zr = zeros (1,20);

  for i = 1:size(z)
    
    if (z(i)>0) 
      
      zr(i) = z(i) ;
    
    end  
  
  end
  
 zr = zr';
  
end