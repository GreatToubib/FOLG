function [ W ] = updateW( W_moins_1, lrW,gdW, W_moins_2 )
%UNTITLED Summary of this function goes here
%   mise a jour des poids 
%   normal: mise a jour classique 
%   heavyball: methode acceleree via une inertie (beta)
%   nesterov: pas implémenté mais meme principe, inertie DANS gdW aussi
    global accel;
    global k;
    global alpha_moins_1;
    if k==1
        alpha_moins_1=0.5;
    end
    if isequal(accel,'normal') 
        W = W_moins_1 - lrW*gdW;     
    end
    if isequal(accel,'heavyball') 
        beta= (k-1)/(k+2); % Paul Tseng scheme
        %alpha = ( sqrt(alpha_moins_1^4+4*alpha_moins_1^2) - alpha_moins_1^2) / 2;
        %beta = alpha_moins_1*(1-alpha_moins_1)/ ( alpha_moins_1^2 + alpha )
        %beta=1; % lazy method, constant beta  
        W = W_moins_1 - lrW*gdW + beta * ( W_moins_1 - W_moins_2 ) ;     
    end
    
    %%%% pas implémenté 
    if isequal(accel,'nesterov') 
        W = W_moins_1 - lrW*gdW;    
    end
end

