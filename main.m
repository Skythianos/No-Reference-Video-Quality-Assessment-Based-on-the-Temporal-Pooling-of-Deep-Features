% Written by: Domonkos Varga

clear all
close all

load KoNViD1k.mat

path = strcat(pwd,filesep,'KoNViD_1k_videos'); % folder where the videos are stored

% Parameters of the algorithm
Constants.BaseArchitecture    = 'inceptionv3'; % 'inceptionv3' or 'inceptionresnetv2' can be choosen
Constants.PoolMethod          = 'avg';         % 'max','min','avg', or 'median' can be choosen
Constants.numberOfVideos      = 1200;          % number of videos in the database
Constants.numberOfTrainVideos = 960;           % number of training videos  
Constants.path                = path;          % path to videos
Constants.useParallelToolbox  = false;         % true or false can be choose (to use or not to use Parallel Computing Toolbox) 
Constants.useTransferLearning = true;          % to use transfer learning or not to use transfer learning (true or false)

% Parameters for transfer learning
ParametersTransferLearning.trainingOptions    = 'sgdm';
ParametersTransferLearning.initialLearnRate   = 1e-4;
ParametersTransferLearning.miniBatchSize      = 32;
ParametersTransferLearning.maxEpochs          = 40;
ParametersTransferLearning.verbose            = false;
ParametersTransferLearning.shuffle            = 'every-epoch';
ParametersTransferLearning.validationPatience = Inf;

[PLCC,SROCC,KROCC] = trainAndTestMethod(Name, MOS, Constants, ParametersTransferLearning);

