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


%vislalize the dataset
visdataset(sigdata,labels)

% Decision Tree Training - simple
% Preprocess Data
% labels = categorical(labels);
% labelVector = grp2idx(labels); % Convert categorical to numeric indices

% Split Data
cv = cvpartition(labels,'HoldOut',0.3);
Xtrain = sigpad(training(cv), :);
Ytrain = labels(training(cv), :);
Xtest = sigpad(test(cv), :);
Ytest = labels(test(cv), :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simple Model
treeModel = fitctree(Xtrain, Ytrain);
view(treeModel, 'Mode', 'graph');

% Evaluation
Ypred = predict(treeModel, Xtest);
confMat = confusionmat(Ytest, Ypred);
accuracy = sum(Ypred == Ytest) / length(Ytest);

% Display Results
disp('Confusion Matrix of Simple Model:');
rowNames = {'Predicted Positive', 'Predicted Negative'};
colNames = {'Actual Positive', 'Actual Negative'};
confMatTable = array2table(confMat, 'RowNames', rowNames, 'VariableNames', colNames);
disp(confMatTable);
disp(['Accuracy (Simple Model): ', num2str(accuracy * 100), '%']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random Forest
treeModel_R = fitensemble(Xtrain, Ytrain, 'Bag', 100, 'Tree', 'Type', 'classification');

Ypred_R = predict(treeModel_R, Xtest);
confMat_R = confusionmat(Ytest, Ypred_R);
accuracy_R= sum(Ypred_R == Ytest) / length(Ytest);

% Display Results
disp('Confusion Matrix of Random Forest:');
rowNames = {'Predicted Positive', 'Predicted Negative'};
colNames = {'Actual Positive', 'Actual Negative'};
confMatTable_R = array2table(confMat_R, 'RowNames', rowNames, 'VariableNames', colNames);
disp(confMatTable_R);
disp(['Accuracy (Random Forest): ', num2str(accuracy_R * 100), '%']);


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




