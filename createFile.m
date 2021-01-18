function [ ] = createFile( Xvr, W_trained, B_trained )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

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