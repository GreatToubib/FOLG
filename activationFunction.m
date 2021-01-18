function [a_ypred, ader ] = activationFunction(ypred, activation)
    % https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
    if isequal(activation,'sigmoid') 
        a_ypred = sigmoidFunction(ypred); 
        % https://kawahara.ca/how-to-compute-the-derivative-of-a-sigmoid-function-fully-worked-example/
        ader = sigmoidFunction(ypred) .* ( 1 - sigmoidFunction(ypred));
        
    elseif isequal(activation,'softmax')
        a_ypred = softmax(ypred) ;
        ader = softmaxder(ypred) ;

    elseif isequal(activation,'relu')
        a_ypred = ReLU (ypred);
        ader = ReLUder (ypred);
    else
        disp('error activationChoice')
    end

end


