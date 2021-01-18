function W = initializationFunction(m, choice)
% https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c

    if isequal(choice,'zeros') %  
        W = zeros (m,20);
    end
    
    if isequal(choice,'random') % random entre -0.5 et 0.5
        W = rand(m,20) ;
        W = -1+W*2;
    end
    if isequal(choice,'xavier') %  le plus utilise avec sigmoid askip
        x = sqrt(6/(1+20)); % 1 input et 20 output par neurone
        W = (rand(m,20) * 2 - 1) * x ;
    end
    if isequal(choice,'he') % he uniform, le + utilise avec relu askip, mais relu foire
        x = sqrt(6/1); % 1 input
        W = (rand(m,20) * 2 - 1) * x ;
    end
    if isnumeric(choice)
        disp('numeric')
        W = ones (m,20) * choice ;
    end

end

