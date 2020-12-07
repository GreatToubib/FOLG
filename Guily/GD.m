function [W] = GD(W, Xts, y_encoded, lr, epoch_number, activation, abs_tol, rel_tol)

for epoch = 1:epoch_number
    epoch
    cost=0;
   
    ypred = W' * Xts ; % ypred avant activation du test sample
    if activation == 'sigmoid'
        a_ypred = sigmoidFunction(ypred);  % ypred avec activation sigmoid
        % https://kawahara.ca/how-to-compute-the-derivative-of-a-sigmoid-function-fully-worked-example/
        ader = sigmoidFunction(ypred) .* ( 1 - sigmoidFunction(ypred));
    end
    %if activation =='softmax'
        %a_ypred = softmax(ypred) ;
        %ader = softmaxder(ypred) ;
    %end
    %if activation =='relu'
        %a_ypred = ReLU (ypred)
        %ader = ReLUder (ypred)
    %end

    cost = cost + costfunction(a_ypred, y_encoded'); % test de la cost function sur le test sample

    part1 = a_ypred-y_encoded';
    part2 = ader;
    part3=  (part1 .*  part2);
    gd = Xts * ( 1/20 * part3)'; 
    
    W = W - lr*gd;
    
    if epoch == 1 
        previous_cost = 10000;
    end

    diff=previous_cost - cost
    if diff  < abs_tol
        disp('breaking abs')
        break
    elseif diff < previous_cost* rel_tol
        disp('breaking rel')
        break
    else
        previous_cost = cost;
      
    end

end
end


	