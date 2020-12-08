function W = initializationFunction(m, choice)
% https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c

    if isequal(choice,'zeros') %  
        W = zeros (m,20);
    end
    
    if isequal(choice,'random') % random avec 0.05 minimum 
        W = rand(m,20) ;
        W = 0.10+W*0.90;
    end
    if isequal(choice,'xavier') % (uniform), le plus utilise avec sigmoid askip
        x = sqrt(6/(1+20)); % 1 inout et 20 output par neurone
        W = (rand(m,20) * 2 - 1) * x ;
    end
    if isequal(choice,'he') % he uniform, le + utilise avec relu askip
        x = sqrt(6/1); % 1 inout
        W = (rand(m,20) * 2 - 1) * x ;
    end
    if isnumeric(choice)
        disp('numeric')
        W = ones (m,20) * choice ;
    end

end

