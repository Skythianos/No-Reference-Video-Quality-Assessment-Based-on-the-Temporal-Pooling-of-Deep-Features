function [PLCC, SROCC, KROCC] = trainAndTestMethod(Name, MOS, Constants, ParametersTransferLearning)

    if(Constants.numberOfTrainVideos>=Constants.numberOfVideos)
        error('Number of training videos cannot be greater than the number of videos'); 
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end

    if(isempty(ParametersTransferLearning) || isempty(ParametersTransferLearning.trainingOptions) || ...
            isempty(ParametersTransferLearning.initialLearnRate) || isempty(ParametersTransferLearning.miniBatchSize) || ...
            isempty(ParametersTransferLearning.maxEpochs) || isempty(ParametersTransferLearning.verbose) || ...
            isempty(ParametersTransferLearning.shuffle) || isempty(ParametersTransferLearning.validationPatience))
        error('Struct ParametersTansferLearning cannot be empty.'); 
    end
    
    if(size(MOS,1)~=size(Name,1))
        error('The variables MOS and Name must have the same length.');
    end
    
    % Dividing database into training and testing datasets
    p = randperm(Constants.numberOfVideos); 
    PermutedName = Name(p);    % random permutation of the videos
    PermutedMOS = MOS(p);      % random permutation of the videos

    AllVideos = PermutedName;
    
    TrainVideos = PermutedName(1:Constants.numberOfTrainVideos);
    TrainMOS    = PermutedMOS(1:Constants.numberOfTrainVideos);

    % TestVideos  = PermutedName((Constants.numberOfTrainVideos+1):end);
    TestMOS     = PermutedMOS((Constants.numberOfTrainVideos+1):end);

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
        createTrainImages(TrainVideos, TrainMOS, Constants);
        if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
            net = transferLearningInceptionV3(net, Constants, ParametersTransferLearning);
        elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
            net = transferLearningInceptionResNetV2(net, Constants, ParametersTransferLearning);
        else
            error('Unknown base architecture'); 
        end
    end

    VideoLevelFeatures = getVideoLevelFeatures(AllVideos, net, Constants);

    TrainVideoLevelFeatures = VideoLevelFeatures(1:Constants.numberOfTrainVideos,:);
    TestVideoLevelFeatures  = VideoLevelFeatures(Constants.numberOfTrainVideos+1:end,:);

    % Training different SVRs (linear, gaussian, 1st order polynomial, 2nd
    % order polynomial, 3rd order polynomial)
    MdlLin = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'linear');
    MdlGan = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
    MdlPoly_1 = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 1);
    MdlPoly_2 = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    MdlPoly_3 = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);

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

end

