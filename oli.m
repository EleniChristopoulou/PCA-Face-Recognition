clear all;
clc;
close all;

load olivetti.mat;
rows = 64;
cols = rows;

%I = X(10,:);
%I = reshape(I, [rows, cols]);
%figure;
%imagesc(I);

% 1. decided to sperate images to 70% training and 30% test

counterTest = 1;
counterTrain = 1;

trainPercent = 70;
testPercent = 100 - trainPercent;

testSubset = zeros(400*testPercent/100,width(X));
trainSubset = zeros(400*trainPercent/100,width(X));

for i = 1:height(X)   %for every photo
    if mod(i, 10) == 8 || mod(i, 10) == 9 || mod(i, 10) == 0
        testSubset(counterTest,:) = X(i,:);
        counterTest = counterTest + 1;
    else
        trainSubset(counterTrain,:) = X(i,:);
        counterTrain = counterTrain + 1;
    end
end

% 2. get the average face of train data
averageFace = zeros(1,width(trainSubset));
for i = 1:height(trainSubset)   %for every photo of train subset
    averageFace = trainSubset(i,:)+ averageFace;
end

averageFace = averageFace / height(trainSubset);
trainSubsetProc = zeros(height(trainSubset),width(trainSubset));    % same dimensions for my processed train matrix

% subtract each face to the average one
for i = 1:height(trainSubset)   %for every photo of train subset        %centered train
    trainSubsetProc(i,:) = trainSubset(i,:) - averageFace;
end

Cov = trainSubsetProc()' * trainSubsetProc(); 
[eigen_vectors, eigen_values] = eig(Cov);
eigen_values = diag(eigen_values);

nrows = 5;      %5x5
ncols = 5;
k = nrows*ncols;
figure; clf; set(gcf, 'Name', 'EigenFaces');
for ii=1:nrows*ncols
 subplot(nrows,ncols,ii);
 eigen_faces(:,ii) = eigen_vectors(:,4097-ii);%.*eigen_values(4097-ii);
 imagesc( reshape(eigen_vectors(:,4097-ii).*eigen_values(4097-ii),rows,cols) );            %eigen faces
 colormap gray; axis equal tight off
end

num = 90;
weights = trainSubsetProc *eigen_faces;
rec= weights(num,:) * eigen_faces'; %+ averageFace*(sum(weights(num,:)));

figure;
subplot(1,2,1);
I = trainSubsetProc(num,:);
I = reshape(I, [rows, cols]);
imagesc(I);
colormap gray;
title('Original Image');

subplot(1,2,2);
% Display the composite image
I = rec;
I = reshape(I, [rows, cols]);
imagesc(I);
colormap gray;
title('Recreation Image');


%store all weights to each person
for i=1:40    %for each person      
    person(i,:,:) = weights(((trainPercent/10)*(i-1)+1):(trainPercent/10)*i,:);      
end

personMeanWeight = zeros(40,k);       %40 10

for i = 1:40
    w = 0;
    for j = 1:(trainPercent/10) 
        w = weights((i-1)*(trainPercent/10)+ j,:) + w;
    end
    personMeanWeight(i,:) = w/k;
end

correct = 0;
false = 0;
real_label = zeros(1,height(testSubset));
prediction = zeros(1,height(testSubset));
for i = 1:height(testSubset)
    weight_test = (testSubset(i,:)-averageFace) *eigen_faces;
    for j = 1:40
        euclidean_distance(j) = pdist([weight_test; personMeanWeight(j,:)], 'euclidean');
    end
    [min_value, min_index] = min(euclidean_distance); % the prediction
    prediction(i) = min_index;
    real_label(i) = ceil(i / 3);    %the one he is
    
    if prediction(i) == real_label(i)
        correct = correct + 1;
    else
        false = false + 1;
    end
end

correct/(false+correct)

% confusion matrix
confusion_matrix = zeros(40);
for i = 1:40
    confusion_matrix(prediction(i),real_label(i)) = confusion_matrix(prediction(i),real_label(i)) + 1;
end
