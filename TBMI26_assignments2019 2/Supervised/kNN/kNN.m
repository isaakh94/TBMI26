function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               Lt - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = sort(unique(Lt));
numClasses = length(classes);

distances = zeros(size(Xt,2),1);
[~, columns] = size(X);
for i = 1:columns
    %Calculating the distance from Xi to each Xtj
    for j = 1: size(Xt,2)
        distances(j) = sqrt(sum((X(:,i)-Xt(:,j)).^2));
    end
    %Sorting the list of training labels according to list of distances
    [~, distances_sorted_order] = sort(distances);
    best_labels = Lt(distances_sorted_order);
    %Picking top k sorted labels
    best_labels = best_labels(1:k);
    %Iterating through the k labels to vote for best result
    votes_for_classes = zeros(numClasses,1);
    for j = 1: length(best_labels)
        votes_for_classes(best_labels(j)) = votes_for_classes(best_labels(j)) + 1;
    end
    %Sorting the classes according to the vote
    [sorted_votes, votes_sorted_order] = sort(votes_for_classes, 'descend');
    tmp = classes(votes_sorted_order);
    %Assigning label
    labelsOut(i) = tmp(1);
    %Handling ties
    if sorted_votes(1) == sorted_votes(2)
        for j = 1:length(best_labels)
            if votes_for_classes(j) == sorted_votes(1)
                labelsOut(i) = best_labels(j);
                break;
            end
        end
    end
end
end

