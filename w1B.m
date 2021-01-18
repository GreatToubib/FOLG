function out=w1B(x,y,d,alpha,beta1,Ytrain, Xtrain)
  % Entrees :
  % x est l'itéré courant qui est un vecteur de taille m+1
  % d est la direction de recherche qui est un vecteur de taille m+1
  % alpha est le pas
  % beta1 est un paramètre

  % Sorties, ce fichier w1.m doit renvoyer :
  % out = 1 si la condition  W1 est vraie
  % out = 0 si la condition  W1 est fausse
  
  falpha = costfunction( Ytrain', 'MSE', x, Xtrain, (y-alpha*d'), 'sigmoid');
  g0   = costfunction( Ytrain', 'MSE', x, Xtrain, y, 'sigmoid');
  
  
  if falpha <= alpha * beta1 * d' * g0
    out = true;
  else 
    out = false;
  end
end