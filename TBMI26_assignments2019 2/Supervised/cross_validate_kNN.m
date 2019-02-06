function [ predictedLabels, trueLabels ] = cross_validate_kNN(numBins, k, X, D, L)
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );
predictedLabels = [];
trueLabels = [];
for i = 1:numBins
    TrainX = [];
    TrainL = [];
    for j = 1:numBins
        if j == i
            continue
        end
        TrainX = [TrainX Xt{j}];
        TrainL = [TrainL Lt{j}];
    end
    predictedLabels = [predictedLabels; kNN(Xt{i}, k, TrainX, TrainL)];
    trueLabels = [trueLabels; Lt{i}];
end
%predictedLabels = predictedLabels.'
%trueLabels = trueLabels.'
end