function [w,epoch_count] = FOLGPerceptron(X,y,w0)
%Linear Classifier

%%Main Function
N = size(X,1);
err = 1;
epoch_count=0;
%poids
w = w0;
while (err > 0) && (epoch_count < 1) % on continue tant que le taux d erreur est > 0 
    err_count=0;
    for row_no = 1:2 % pour chaque data point, 
        %activation
        ypred = softmax(X(:,row_no)*w0);
        ypred = softmax(ypred);
        %y(row_no,:)
        error = immse(y(row_no,:), ypred)
        
        
        %back propagation
        %weights actualization
        
        % error count actualization
        err_count=err_count+1;
    end
    epoch_count=epoch_count+1;
    err = err_count/N % taux d erreur
   
end



