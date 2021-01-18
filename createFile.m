function [ ] = createFile( Xvr, W_trained, B_trained )
% create submissions File for Kaggle

[m,p] = size(Xvr);
data = zeros(p,2);
size(data)
for i = 1:p
    [m,p] = size(Xvr);
    ypred = W_trained' * Xvr (:,i) + B_trained ; 
    a_ypred = sigmoidFunction(ypred);
    [M,I] = max(a_ypred);
    data(i,1) = i;
    data(i,2) = I;
end
csvwrite('submissions.csv',data);
end