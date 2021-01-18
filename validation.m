function [accuracy ] = validation( Xtest, Ytest, W_trained, B_trained )
%UNTITLED6 Summary of this function goes here
%   Validation d'un modele(W_trained,B_trained) sur le validation set, 
%   OUTPUT: accuracy. 
count=0;
[m,p] = size(Xtest);
for test_sample = 1:p
    [m,p] = size(Xtest);
    ypred = W_trained' * Xtest (:,test_sample) + B_trained ; 
    a_ypred = sigmoidFunction(ypred);
    [M,I] = max(a_ypred);
    ytrue = Ytest(test_sample);
    if ytrue== I 
        count= count+1;
    end
end
accuracy = count/p

end

