clear;
clc;
sourceDir = 'C:\Thomas\UNSW\thesis\data\Training data v2\Trainingdata';
targetDir = "D:\UNSW\Thesis training\Scalogram regression_v3\Step1_generation\test";

% Get a list of all .mat files in sourceDir
matFiles = dir(fullfile(sourceDir, '*.mat'));

% Define fixed range for cfs magnitude
cfs_min = 0;      % Minimum magnitude
cfs_max = 1.52;   % Maximum magnitude (based on your observed max)

% Loop through each .mat file found
for i = 1:length(matFiles)
    fileName = matFiles(i).name;
    folderPath = fullfile(targetDir, erase(fileName, '.mat'));
    
    % Create folder if it does not exist
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
    
    % Load the .mat file
    filePath = fullfile(sourceDir, fileName);
    data = load(filePath);
    
    % Get the variable name dynamically
    varName = fieldnames(data);
    signals = data.(varName{1}); % Assuming one variable per .mat file
    
    % Loop through each signal
    for j = 1:length(signals)
        signal = signals{j};
        signal = signal(:); % Convert to column vector
        
        % Add 500 samples of 0 to the start and end
        signal = [zeros(500, 1); signal; zeros(500, 1)];
        
        % Sampling frequency
        fs = 3.125e9;
        
        % Compute CWT scalogram
        [cfs, ~] = cwt(signal, fs, 'morse', 'VoicesPerOctave', 32, 'FrequencyLimits', [0 1000e6]);  
        cfs_mag = abs(cfs);  % Use absolute magnitude
        
        % Clamp cfs magnitude to the range [0, 1.55]
        cfs_mag = max(min(cfs_mag, cfs_max), cfs_min);
        
        % Convert to RGB using a perceptually uniform colormap (turbo/hot)
        colormap_used = turbo(256);  % Turbo provides better contrast
        cfs_RGB = ind2rgb(round((cfs_mag / cfs_max) * 255) + 1, colormap_used);
        
        % Resize while maintaining aspect ratio
        intermediateWidth = max(round(size(cfs_RGB, 2) / 2), 1500);
        cfs_RGB_resized = imresize(imresize(cfs_RGB, [256, intermediateWidth], 'lanczos3'), [256, 512], 'lanczos3');
        
        % Save as PNG
        saveFileName = fullfile(folderPath, sprintf('%s_%d.png', erase(fileName, '.mat'), j));
        imwrite(cfs_RGB_resized, saveFileName);
    end
end

disp('Scalograms for all .mat files generated successfully.');
