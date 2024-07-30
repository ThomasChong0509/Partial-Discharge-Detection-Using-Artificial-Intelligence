clc;
clear;
close all;

% Create label
pathToSignals = "C:\Thomas\UNSW\thesis\data\data500ns2class";
sigds = signalDatastore(pathToSignals, "IncludeSubfolders",true,"FileExtensions",".mat");
labels = folders2labels(pathToSignals);

lbnum = transform(sigds, @callbs);
lbnum = readall(lbnum);
labels = lbexpension(lbnum,labels);
sigt = transform(sigds,@prepsig);
sigdata = readall(sigt);
sigpad = cell2mat(sigdata);
labels = categorical(labels); % Convert labels to categorical

% Assuming each sequence in 'sigdata' is a 1D sequence of size (1xN)
% Reshape into a 2D sequence where each row is a time step (Nx1)
numSamples = numel(sigdata);
seqLength = size(sigdata{1}, 2); % Assuming each sequence is of same length

% Initialize cell array to store sequences
X = cell(numSamples, 1);
for i = 1:numSamples
    X{i} = reshape(sigdata{i}, [seqLength, 1]); % Reshape to (seqLength x 1)
end

%vislalize the dataset
visdataset(sigdata,labels)

% Decision Tree Training - simple
% Preprocess Data
% labels = categorical(labels);
% labelVector = grp2idx(labels); % Convert categorical to numeric indices

% Split Data
cv = cvpartition(labels,'HoldOut',0.3);
Xtrain = X(training(cv), :);
Ytrain = labels(training(cv), :);
Xtest = X(test(cv), :);
Ytest = labels(test(cv), :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
% Define a range of hidden sizes to test
hiddenSizes = [2 10 20 50];
accuracies = zeros(size(hiddenSizes));

% Model
for i = 1:length(hiddenSizes)
    hiddenSize = hiddenSizes(i);
    % Define the LTSM layers
    layers = [ ...
    sequenceInputLayer(size(sigpad,2))
    lstmLayer(hiddenSize, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    sigmoidLayer
    classificationLayer];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 1e-6, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');
    
    % Train the RNN
    net = trainNetwork(Xtrain, Ytrain, layers, options);
    
    % Predict the labels for the test set
    Ypred = classify(net, Xtest);
    accuracy = sum(Ypred == Ytest) / numel(Ytest);
    accuracies(i) = accuracy;
    fprintf('Hidden Size: %d, Test Accuracy: %.2f%%\n', hiddenSize, accuracy * 100);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function
function sig = prepsig(sigin)
sig = sigin(:,:); % Select the first 4000 columns
sig = num2cell(sig,2); % Convert each row to a cell
end

function lbnum = callbs(sigin)
lbnum = size(sigin,1);
end

function newlb = lbexpension(lbN,lbs)
a = 1;
for i = 1:length(lbN)
    for j = 1:lbN(i)
        newlb(a,1) = lbs(i);
        a = a+1;
    end
end
end

function visdataset(data,labels)
    figure
    sampling_rate = 3.125e9;
    tiledlayout(4,4)
    for k = 1:16
        n = 104*k;
        nexttile
        num_samples = size(data{n},2);
        time_seconds = (0:num_samples-1) * (1/sampling_rate);
        plot(time_seconds,data{n})
        title(labels(n))
        xlabel('Time (seconds)');
        ylabel('Normalized Value');
    end
end
function wrongprediction=findpredind(testlabels,testpred)
for i = 1:length(testlabels)
    if testpred(i)~=testlabels(i)
        wrongprediction(i) = i;
    end
end
wrongprediction(find(wrongprediction==0))=[];
end
