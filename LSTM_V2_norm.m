clc;
clear;
close all;

% Create label
pathToSignals = "C:\Thomas\UNSW\thesis\data\data500ns2class";
sigds = signalDatastore(pathToSignals, "IncludeSubfolders",true,"FileExtensions",".mat");
labels = folders2labels(pathToSignals);

start_time = 500e-9;  % 500 ns
sampling_rate = 3.125e9;  % 3.125 GHz
start_index = round(start_time * sampling_rate);  % Convert time to index

lbnum = transform(sigds, @callbs);
lbnum = readall(lbnum);
labels = lbexpension(lbnum,labels);
sigt = transform(sigds,@prepsig);
sigdata = readall(sigt);
sigpad = cell2mat(sigdata);
sigpad = sigpad(:,:);

row_sizes = ones(size(sigpad, 1), 1);  % One row per cell
col_sizes = size(sigpad, 2);  % Keep all columns in each cell
sigdata = mat2cell(sigpad, row_sizes, col_sizes);

labels = categorical(labels); % Convert labels to categorical

visdataset(sigdata,labels)

% Assuming each sequence in 'sigdata' is a 1D sequence of size (1xN)
% Reshape into a 2D sequence where each row is a time step (Nx1)
numSamples = numel(sigdata);
seqLength = size(sigdata{1}, 2); % Assuming each sequence is of same length

% Initialize cell array to store sequences
X = cell(numSamples, 1);
for i = 1:numSamples
    X{i} = reshape(sigdata{i}, [seqLength, 1]); % Reshape to (seqLength x 1)
end


% Decision Tree Training - simple
% Preprocess Data
% labels = categorical(labels);
% labelVector = grp2idx(labels); % Convert categorical to numeric indices

% Step 1: Split into 85% (training + cross-validation) and 15% (testing)
cv1 = cvpartition(labels, 'HoldOut', 0.15); 

% Extract training+cross-validation and testing sets
XtrainCV = X(training(cv1), :);
YtrainCV = labels(training(cv1), :);
Xtest = X(test(cv1), :);
Ytest = labels(test(cv1), :);

% Step 2: Split 85% data (training + cross-validation) into 70% (training) and 15% (cross-validation)
cv2 = cvpartition(YtrainCV, 'HoldOut', 0.1765); % 15/85 ≈ 0.1765

% Extract training and cross-validation sets
Xtrain = XtrainCV(training(cv2), :);
Ytrain = YtrainCV(training(cv2), :);
Xval = XtrainCV(test(cv2), :);
Yval = YtrainCV(test(cv2), :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
% Define a range of hidden sizes to test
hiddenSizes = [16];
accuracies = zeros(size(hiddenSizes));

% Model
for i = 1:length(hiddenSizes)
    hiddenSize = hiddenSizes(i);

    % Define the LTSM layers
     layers = [ ...
        sequenceInputLayer(size(sigpad,2))           % Input layer
        lstmLayer(hiddenSize, 'OutputMode', 'last')    % Scaled hidden size
        dropoutLayer(0.5)
        fullyConnectedLayer(2)                       % Output layer
        softmaxLayer                                 % Softmax activation for classification
        classificationLayer];                        % Classification layer
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'ValidationData',{Xval,Yval}, ...
        'ValidationFrequency',50, ...
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
    sig = sigin(:,:); % Select all columns
    % Perform min-max normalization (regular normalization)
    %sig = (sig - min(sig, [], 2)) ./ (max(sig, [], 2) - min(sig, [], 2));
    sig = normalize(sig);
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
    num_data = length(data);  % Total number of datasets
    num_plots = min(16, num_data);  % Plot up to 16 datasets or less if num_data < 16
    
    tiledlayout(4,4)
    
    for k = 1:num_plots
        n = round((k-1) * num_data / num_plots) + 1;  % Dynamically pick dataset index
        nexttile
        
        num_samples = size(data{n}, 2);
        time_seconds = (0:num_samples-1) * (1/sampling_rate);
        plot(time_seconds, data{n})
        
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
