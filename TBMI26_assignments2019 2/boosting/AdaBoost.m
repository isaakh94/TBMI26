%% Hyper-parameters
%  You will need to change these. Start with a small number and increase
%  when your algorithm is working.

% Number of randomized Haar-features
nbrHaarFeatures = 75;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;
% Number of weak classifiers
nbrWeakClassifiers = 250;

%% Load face and non-face data and plot a few examples
%  Note that the data sets are shuffled each time you run the script.
%  This is to prevent a solution that is tailored to specific images.

load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do NOT modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
A = zeros(nbrWeakClassifiers,1); %WeakClassifier weights
T = zeros(nbrWeakClassifiers,1); %Thresholds
P = zeros(nbrWeakClassifiers,1); %Planarities
F = zeros(nbrWeakClassifiers,1); %Which features to cut on
D = ones(size(xTrain,2),1)/size(xTrain,2); %Weights of samples
planarity = 1;

for iteration = 1:nbrWeakClassifiers
    disp(iteration)
    minerror = inf;
    for feature = 1:nbrHaarFeatures
        thresholds = unique(xTrain(feature,:));
        for threshold = 1:length(thresholds)
            C = WeakClassifier(thresholds(threshold), planarity, xTrain(feature,:));
            error = WeakClassifierError(C, D, yTrain);
            if error > 0.5
                error = 1 - error;
                planarity = -planarity;
            end
            if error < minerror
                T(iteration) = thresholds(threshold);
                P(iteration) = planarity;
                F(iteration) = feature;
                minerror = error;
            end
        end
    end
    A(iteration) = (1/2)*log((1-minerror)/minerror);
    C = WeakClassifier(T(iteration), P(iteration), xTrain(F(iteration),:));
    D = D.*(exp(-A(iteration)*yTrain.*C))';
    D = D/sum(D);
end

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.
accTrain = zeros(nbrWeakClassifiers,1);
accTest = zeros(nbrWeakClassifiers,1);

weakclassTrain = zeros(nbrWeakClassifiers,size(xTrain,2));
strongclassTrain = zeros(nbrWeakClassifiers,size(xTrain,2));
weakclassTest = zeros(nbrWeakClassifiers,size(xTest,2));
strongclassTest = zeros(nbrWeakClassifiers,size(xTest,2));
for iteration = 1:nbrWeakClassifiers
    weakclassTrain(iteration,:) = WeakClassifier(T(iteration), P(iteration), xTrain(F(iteration),:));
    strongclassTrain(iteration,:) = sign(sum(weakclassTrain(1:iteration,:).*A(1:iteration), 1));
    accTrain(iteration) = sum(strongclassTrain(iteration,:) == yTrain)/length(yTrain);
    weakclassTest(iteration,:) = WeakClassifier(T(iteration), P(iteration), xTest(F(iteration),:));
    strongclassTest(iteration,:) = sign(sum(weakclassTest(1:iteration,:).*A(1:iteration), 1)); 
    accTest(iteration) = sum(strongclassTest(iteration,:) == yTest)/length(yTest);
end

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
plot(1:nbrWeakClassifiers,accTrain)
hold on
plot(1:nbrWeakClassifiers,accTest)
legend('Train', 'Test', 'Location', 'southeast')
xlabel('Number of weak classifiers')
ylabel('Accuracy')
hold off

%% Plot some of the misclassified faces and non-faces from the test set
%  Use the subplot command to make nice figures with multiple images.
missclassified_faces = find(strongclassTest(70,:) ~= yTest & yTest == 1);
missclassified_other = find(strongclassTest(70,:) ~= yTest & yTest == -1);

cherry_picked_faces = missclassified_faces([1 3 6 24 9 23 27 35 16 18 22 28 7 57 58 47]);
cherry_picked_other = missclassified_other([63 59 58 55 33 38 42 17 1 9 13 35]);

figure(4);
colormap gray;
for k=1:12
    subplot(3,4,k), imagesc(testImages(:,:,cherry_picked_other(k)));
    axis image;
    axis off;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
features_used = unique(F(1:70))

figure(5);
colormap gray;
for k = 1:length(features_used)
    subplot(5,9,k),imagesc(haarFeatureMasks(:,:,features_used(k)),[-1 2]);
    title(sum(F(1:70) == features_used(k)));
    axis image;
    axis off;
end