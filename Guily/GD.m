function [W] = GD(W, Xts, y_encoded, lr, epoch_number, activation, abs_tol)

for epoch = 1:epoch_number
    epoch
    cost=0;
    for i = 1:13960
        ypred = W * Xts (:,i) ; % ypred avant activation du test sample

        if activation =='sigmoid'
            a_ypred = sigmoid(ypred);  % ypred avec activation sigmoid
            ader = sigmoidder (ypred);
        end
        if activation =='softmax'
            a_ypred = softmax(ypred) ;
            ader = softmaxder(ypred) ;
        end
        %if activation =='relu'
            %a_ypred = ReLU (ypred)
            %ader = ReLUder (ypred)
        %end
        
        cost = cost + costfunction(a_ypred, y_encoded (i,:)); % test de la cost function sur le test sample
        
        
       

    end
    if epoch == 1 
        previous_cost = 10000;
    end
    previous_cost
    cost=cost/13960
    
    if previous_cost - cost  < abs_tol
        disp('breaking')
        break
    else
        previous_cost = cost;
        W = W - lr*ader;
    end
end
end
	