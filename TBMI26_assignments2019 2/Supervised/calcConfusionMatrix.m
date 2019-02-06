function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

% Add your own code here
% Assumes all classes belong to range 1, 2, ..., n
for i = 1:size(Lclass,1)
    cM(Lclass(i),Ltrue(i)) = cM(Lclass(i),Ltrue(i)) + 1;
end
end

