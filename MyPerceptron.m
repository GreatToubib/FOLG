function [w,epoch_count] = MyPerceptron(X,y,w0)
%Linear Classifier

%%Visual Part Code
sz= 15;
a = X(:,1);
b = -w0(1)*a/w0(2);
hold on
for i = 1:size(y)
    if(y(i)==1)
        scatter(X(i,1),X(i,2),sz,'b','filled')
    else
        scatter(X(i,1),X(i,2),sz,'r','filled')
    end
end
plot(a,b,'k');
axis([-1.5 1.5 -1.5 1.5]);
hold off

%%Main Function
N = size(X,1);
err = 1;
epoch_count=0;
%poids
w = w0;
while err >0 % on continue tant que le taux d erreur est > 0 
    err_count=0;
    for row_no = 1: N % pour chaque data point, 
        %activation
        if sign(X(row_no,:)*w) ~= y(row_no) % test avant backpropagation, si incorrect, corriger.
            %Y = sign(x) returns an array Y the same size as x, where each element of Y is:
            %1 if the corresponding element of x is greater than 0.
            %0 if the corresponding element of x equals 0.
            %-1 if the corresponding element of x is less than 0.
            
            % actualisation des poids: 
            w=w + 1 * X(row_no,:)'*y(row_no); 
            
            % actualisation du compteur d erreurs
            err_count=err_count+1;
        end
    end
    epoch_count=epoch_count+1;
    err = err_count/N % taux d erreur 
end



