load('data1.mat');
[m,p] = size(Xts);
w0 = ones(20,m)*0.01;

%one-hot encoding
y_encoded = (yts==1:20);

[w,epoch_count] = FOLGPerceptron(Xts,y_encoded,w0);


