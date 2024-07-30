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


% Elbow Plot
wcss = [];
for k = 1:10
    [idx1,C1,s] = kmeans(sigpad,k);
    wcss(k)=sum(s);
end

figure
plot (1:k, wcss);
xlabel ('Number of clusters (K)');
ylabel ('WCSS');
title ('Elbow Plot');


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





