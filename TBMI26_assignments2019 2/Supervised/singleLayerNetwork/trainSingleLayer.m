function [Wout, trainingError, testError ] = trainSingleLayer(Xt,Dt,Xtest,Dtest, W0,numIterations, learningRate )
%TRAINSINGLELAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the neurons (matrix)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
Nt = size(Xt,2);
Ntest = size(Xtest,2);
Wout = W0;

% Calculate initial error
Yt = runSingleLayer(Xt, W0);
Ytest = runSingleLayer(Xtest, W0);
trainingError(1) = sum(sum((Yt - Dt).^2))/Nt;
testError(1) = sum(sum((Ytest - Dtest).^2))/Ntest;

for n = 1:numIterations
    
    Y = runSingleLayer(Xt, Wout);
    %disp(size((Y - Dt).*(1-Y.^2)))
    %disp(size(Xt))
    %for j = 1:size(Wout, 1)
    %    grad_w(j,:) = (Y(j,:) - Dt(j,:)).*(1-Y(j,:).^2)*Xt.'
    %end
    grad_w = (Y - Dt)*Xt';

    Wout = Wout - learningRate*grad_w/Nt;
    trainingError(1+n) = sum(sum((runSingleLayer(Xt, Wout) - Dt).^2))/Nt;
    testError(1+n) = sum(sum((runSingleLayer(Xtest, Wout) - Dtest).^2))/Ntest;
end
end

