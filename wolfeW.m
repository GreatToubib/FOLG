function alpha=wolfe(x,y,d,alpha0,Ytrain, Xtrain)
    % Entrees :
    % x est l'itéré courant qui est un vecteur de taille m+1
    % d est la direction de recherche qui est un vecteur de taille m+1
    % alpha0 est le pas initial
    % beta1 est un paramètre
    % beta2 est un paramètre
    % lambda est un paramètre

    % Sorties, ce fichier wolfe.m doit renvoyer :
    % alpha, un pas vérifiant les conditions de Wolfe

    % Parametres pour les conditions de Wolfe
    beta1  = 0.1;
    beta2  = 0.9;
    lambda = 2;

    aleft  = 0;
    aright = inf;
    alpha  = alpha0;
    while 1
        if w1(x,y,d,alpha,beta1,Ytrain, Xtrain)==true && w2(x,y,d,alpha,beta2,Ytrain, Xtrain)==true
            break;
        end
        if w1(x,y, d,alpha,beta1,Ytrain, Xtrain)==false
            % si la premiere condition est fausse, on diminue alpha au
            % milieu de ses bornes
            aright = alpha;
            alpha_old = alpha;
            alpha = (aright+aleft)/2;
            if alpha-alpha_old < 1e-15
                break
            end
            
        elseif w2(x,y,d,alpha,beta2,Ytrain, Xtrain)==false
            % si w2 not ok, on augmente alpha. 
            aleft = alpha;
            alpha_old = alpha;
            if aright < inf
                alpha = (aright+aleft)/2;
            else
                alpha = lambda*alpha;
            end
            if alpha-alpha_old < 1e-15
                break
            end
            
        end
     end
end