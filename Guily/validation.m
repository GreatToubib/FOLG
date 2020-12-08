function [accuracy ] = validation( Xtest, Ytest, W_trained )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    % validation 
count=0;
[m,p] = size(Xtest);
for test_sample = 1:p
    [m,p] = size(Xtest);
    ypred = W_trained' * Xtest (:,test_sample) ; 
    a_ypred = sigmoidFunction(ypred);
    [M,I] = max(a_ypred);
    ytrue = Ytest(test_sample);
    if ytrue== I 
        count= count+1;
    end
end
accuracy = count/p

end

