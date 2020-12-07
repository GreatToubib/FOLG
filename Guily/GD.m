function [W] = GD(W, Xts, y_encoded, lr, epoch_number, activation, abs_tol)

for epoch = 1:epoch_number
    epoch
    cost=0;
    for i = 1:13960
        Xts_row = Xts(:,i);
        ytrue = y_encoded (i,:);
        ypred = W * Xts_row ; % ypred avant activation du test sample
        size(ypred)
        if activation =='sigmoid'
            a_ypred = sigmoid(ypred);  % ypred avec activation sigmoid
            ader = sigmoidder (ypred);
        end
        %if activation =='softmax'
            %a_ypred = softmax(ypred) ;
            %ader = softmaxder(ypred) ;
        %end
        %if activation =='relu'
            %a_ypred = ReLU (ypred)
            %ader = ReLUder (ypred)
        %end
        size(a_ypred)
        
        cost = cost + costfunction(a_ypred, ytrue); % test de la cost function sur le test sample
        
        part1 = a_ypred-ytrue;
        size(part1)
        part2 = ader;
        size(part2)
        part3=  (part1' .*  part2);
        size(part3)
        Xts_row=Xts_row';
        size(Xts_row)
        gd = Xts_row * ( 1/20 * part3)';
        W = W - lr*gd;
       
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
        
    end
end
end


	