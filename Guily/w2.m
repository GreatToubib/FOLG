function out=w2(x,y,d,alpha,beta2,Ytrain, Xtrain)
  % Entrees :
  % x est l'itéré courant qui est un vecteur de taille m+1
  % d est la direction de recherche qui est un vecteur de taille m+1
  % alpha est le pas
  % beta2 est un paramètre

  % Sorties, ce fichier w2.m doit renvoyer :
  % out = 1 si la condition  W2 est vraie
  % out = 0 si la condition  W2 est fausse
  
  galpha   = costfunction( Ytrain', 'MSE', (x-alpha*d), Xtrain, y, 'sigmoid');
  g0   = costfunction( Ytrain', 'MSE', x, Xtrain, y, 'sigmoid');
  
  if d' * galpha >= beta2 * d' * g0 
    out = true;
  else 
    out = false;
  end
end