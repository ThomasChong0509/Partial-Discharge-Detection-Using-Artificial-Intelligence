clc; clear; close all;

%% Get Script Directory (Portable Path)
if isdeployed
    scriptDir = pwd; % For deployed apps
else
    scriptDir = fileparts(matlab.desktop.editor.getActiveFilename()); % For normal scripts
end

% Define Base Paths
basePath = fullfile(scriptDir,'Step1_generation', 'Step1_resultv3'); 
modelDir = fullfile(scriptDir, 'Model'); % Folder to save the model

% Ensure model directory exists
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

%% Define Paths (Relative)
trainDataPath = fullfile(basePath, 'trainData');
trainLabelPath = fullfile(basePath, 'trainLabels');
valDataPath = fullfile(basePath, 'valData');
valLabelPath = fullfile(basePath, 'valLabels');
modelSavePath = fullfile(modelDir, 'cnn_autoencoder.mat'); % Model path
plotSavePath = fullfile(modelDir, 'training_progress.png'); % Plot path

%% Load Training Data
trainImages = imageDatastore(trainDataPath, 'FileExtensions', '.png', 'ReadFcn', @customReadImage);
trainLabels = imageDatastore(trainLabelPath, 'FileExtensions', '.png', 'ReadFcn', @customReadImage);
valImages = imageDatastore(valDataPath, 'FileExtensions', '.png', 'ReadFcn', @customReadImage);
valLabels = imageDatastore(valLabelPath, 'FileExtensions', '.png', 'ReadFcn', @customReadImage);

% Combine input and labels into one datastore
trainData = combine(trainImages, trainLabels);
valData = combine(valImages, valLabels);

%% Define CNN Autoencoder Architecture
layers = [
    imageInputLayer([256 512 3], 'Normalization', 'none')

    % ===== ENCODER =====
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % [128 × 256]

    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % [64 × 128]

    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % [32 × 64]

    % ===== BOTTLENECK =====
    fullyConnectedLayer(1500)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(32*64*64)
    reluLayer
    reshapeLayer([32 64 64])

    % ===== DECODER =====
    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', [1 1]) % Fix size to [64 × 128]
    reluLayer

    transposedConv2dLayer(4, 16, 'Stride', 2, 'Cropping', [1 1]) % Fix size to [128 × 256]
    reluLayer

    transposedConv2dLayer(4, 3, 'Stride', 2, 'Cropping', [1 1]) % Fix size to [256 × 512]
    sigmoidLayer
    regressionLayer
];

%% Training Options with Custom Learning Rate Function
% Define Learning Rate Drop Strategy (Piecewise)
initialLR = 1e-3;   % Starting learning rate
dropFactor = 0.5;   % Reduce by 50%
dropEpochs = [15, 30, 100]; % Epochs where LR drops

% Define Training Options with Piecewise Learning Rate
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', initialLR, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'L2Regularization', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...  % Use piecewise instead of custom
    'LearnRateDropFactor', dropFactor, ... % Reduce LR by 50% at dropEpochs
    'LearnRateDropPeriod', dropEpochs(1), ... % Drop period (only one value allowed)
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputNetwork', 'best-validation'); % Saves the best model


%% Train the Model
net = trainNetwork(trainData, layers, options);

%% Save Model
save(modelSavePath, 'net');

%% Save Plot
fig = findall(groot, 'Type', 'Figure', 'Name', 'Training Progress');
if ~isempty(fig)
    saveas(fig, plotSavePath); % Save as PNG
end

disp("Training complete. Model saved!");

%% Helper Function: Read Image (Normalize to [0,1])
function img = customReadImage(filename)
    img = imread(filename);
    img = im2double(img); % Convert uint8 [0,255] to double [0,1]
end
