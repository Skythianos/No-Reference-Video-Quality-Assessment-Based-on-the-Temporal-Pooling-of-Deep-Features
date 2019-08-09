clear all
close all

load KoNViD1k.mat % video names and MOS

path = strcat(pwd,filesep,'KoNViD_1k_videos'); % folder where the videos are stored

% Parameters of the algorithm
Constants.BaseArchitecture    = 'inceptionv3'; % 'inceptionv3' or 'inceptionresnetv2' can be choosen
Constants.PoolMethod          = 'avg';         % 'max','min','avg', or 'median' can be choosen
Constants.numberOfVideos      = 1200;          % number of videos in the database
Constants.numberOfTrainVideos = 720;           % number of training videos
Constants.numberOfValidationVideos = 240;      % number of validation videos
Constants.path                = path;          % path to videos
Constants.useParallelToolbox  = true;          % true or false can be choose (to use or not to use Parallel Computing Toolbox) 
Constants.useTransferLearning = true;          % to use transfer learning or not to use transfer learning (true or false)

c=parcluster;
c.NumWorkers=6;
saveProfile(c);

delete(gcp('nocreate'));
parpool('local',6);

% Parameters for transfer learning
ParametersTransferLearning.trainingOptions    = 'sgdm';        % adam
ParametersTransferLearning.initialLearnRate   = 1e-4;          % 1e-5 
ParametersTransferLearning.miniBatchSize      = 28;            % 32
ParametersTransferLearning.maxEpochs          = 40;            % 100
ParametersTransferLearning.verbose            = false;
ParametersTransferLearning.shuffle            = 'every-epoch';
ParametersTransferLearning.validationPatience = Inf;
ParametersTransferLearning.N                  = 3;             % stops network
                                                               % training if the best
                                                               % classification accuracy on the validation
                                                               % data does not improve for N network
                                                               % validations in a row.


% Dividing database into training and testing datasets
p = randperm(Constants.numberOfVideos); 
PermutedName = Name(p);    % random permutation of the videos
PermutedMOS = MOS(p);      % random permutation of the videos

AllVideos = PermutedName;
    
TrainVideos = PermutedName(1:Constants.numberOfTrainVideos);
TrainMOS    = PermutedMOS(1:Constants.numberOfTrainVideos);
    
ValidationVideos = PermutedName(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
ValidationMOS    = PermutedMOS(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);

% TestVideos  = PermutedName((Constants.numberOfTrainVideos+1):end);
TestMOS     = PermutedMOS((Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1):end);

% Loading pretrained CNN
if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
    net = inceptionv3;
elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
    net = inceptionresnetv2;
else
    error('Unknown base architecture');
end

% Transfer learning
if(Constants.useTransferLearning)
        createTrainImages(TrainVideos, ValidationVideos, TrainMOS, ValidationMOS, Constants);
        if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
            
            path = strrep(Constants.path, 'KoNViD_1k_videos', '');
            
            trainFrames = imageDatastore(strcat(path,filesep,'SortedFramesTrain'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            valFrames   = imageDatastore(strcat(path,filesep,'SortedFramesValidation'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

            numClasses = numel(categories(trainFrames.Labels));

            lgraph = layerGraph(net);

            lgraph = removeLayers(lgraph, {'predictions_softmax','ClassificationLayer_predictions'});
            newLayers = [
                fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
                softmaxLayer('Name','softmax')
                classificationLayer('Name','classoutput')];
            
            lgraph = addLayers(lgraph, newLayers);
            lgraph = connectLayers(lgraph,'predictions','fc');

            numIterationsPerEpoch = floor(numel(trainFrames.Labels)/ParametersTransferLearning.miniBatchSize);
            options = trainingOptions(ParametersTransferLearning.trainingOptions,'MiniBatchSize',...
                ParametersTransferLearning.miniBatchSize,'MaxEpochs',ParametersTransferLearning.maxEpochs,...
                'InitialLearnRate',ParametersTransferLearning.initialLearnRate,'Verbose',...
                ParametersTransferLearning.verbose,'Plots','training-progress',...
                'ValidationData',valFrames,'ValidationFrequency',numIterationsPerEpoch,...
                'ValidationPatience',ParametersTransferLearning.validationPatience,'Shuffle',...
                ParametersTransferLearning.shuffle,...
                'ExecutionEnvironment','gpu',...
                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,ParametersTransferLearning.N));

            net = trainNetwork(trainFrames, lgraph, options);
            
            elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
            
            path = strrep(Constants.path, 'KoNViD_1k_videos', '');
            
            trainFrames = imageDatastore(strcat(path,filesep,'SortedFramesTrain'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            valFrames   = imageDatastore(strcat(path,filesep,'SortedFramesValidation'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

            numClasses = numel(categories(trainFrames.Labels));
            
            lgraph = layerGraph(net);

            lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});
            newLayers = [
                fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
                softmaxLayer('Name','softmax')
                classificationLayer('Name','classoutput')];
        
            lgraph = addLayers(lgraph, newLayers);
            lgraph = connectLayers(lgraph,'avg_pool','fc');

            numIterationsPerEpoch = floor(numel(train.Labels)/miniBatchSize);
            options = trainingOptions(ParametersTransferLearning.trainingOptions,'MiniBatchSize',...
                ParametersTransferLearning.miniBatchSize,'MaxEpochs',ParametersTransferLearning.maxEpochs,...
                'InitialLearnRate',ParametersTransferLearning.initialLearnRate,'Verbose',...
                ParametersTransferLearning.verbose,'Plots','training-progress',...
                'ValidationData',valFrames,'ValidationFrequency',numIterationsPerEpoch,'ValidationPatience',...
                ParametersTransferLearning.validationPatience,'Shuffle',ParametersTransferLearning.shuffle,...
                'ExecutionEnvironment','gpu',...
                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,ParametersTransferLearning.N));

            net = trainNetwork(trainFrames, lgraph, options);
            
else
            error('Unknown base architecture'); 
        end
end

VideoLevelFeatures = getVideoLevelFeatures(AllVideos, net, Constants);
TrainVideoLevelFeatures = VideoLevelFeatures(1:(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos),:);
TestVideoLevelFeatures  = VideoLevelFeatures((Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1):end,:);

% Training different SVRs (linear, gaussian, 1st order polynomial, 2nd
% order polynomial, 3rd order polynomial)
MdlLin = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'linear');
MdlGan = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
MdlPoly_1 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 1);
MdlPoly_2 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
MdlPoly_3 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);

% Testing SVRs (linear, gaussian, 1st order polynomial, 2nd order
% polynomial, 3rd order polynomial)
YPredLin = predict(MdlLin, TestVideoLevelFeatures);
YPredGan = predict(MdlGan, TestVideoLevelFeatures);
YPredPoly_1 = predict(MdlPoly_1, TestVideoLevelFeatures);
YPredPoly_2 = predict(MdlPoly_2, TestVideoLevelFeatures);
YPredPoly_3 = predict(MdlPoly_3, TestVideoLevelFeatures);

PLCC.Linear = corr(YPredLin, TestMOS, 'Type', 'Pearson');
SROCC.Linear= corr(YPredLin, TestMOS, 'Type', 'Spearman');
KROCC.Linear= corr(YPredLin, TestMOS, 'Type', 'Kendall');

PLCC.Gaussian = corr(YPredGan, TestMOS, 'Type', 'Pearson');
SROCC.Gaussian= corr(YPredGan, TestMOS, 'Type', 'Spearman');
KROCC.Gaussian= corr(YPredGan, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_1 = corr(YPredPoly_1, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_1= corr(YPredPoly_1, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_1= corr(YPredPoly_1, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_2 = corr(YPredPoly_2, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_2= corr(YPredPoly_2, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_2= corr(YPredPoly_2, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_3 = corr(YPredPoly_3, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_3= corr(YPredPoly_3, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_3= corr(YPredPoly_3, TestMOS, 'Type', 'Kendall');
