function [VideoLevelFeatures] = extractVideoLevelFeatures(video, net, Constants)
    % This function extracts the video-level features    
    
    if(~isa(net,'DAGNetwork'))
        error('Variable net must be DAGNetwork');
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    
    % Extracting all frame-level features of the video sequence
    k=1;
    FrameLevelFeatures = zeros( ceil(video.FrameRate*video.Duration), 2048);
    while hasFrame(video)
        frame = readFrame(video);
        FrameLevelFeatures(k,:) = extractFrameLevelFeatures(frame, net, Constants);
        k=k+1;
    end
    FrameLevelFeatures = FrameLevelFeatures(1:(k-1), :);
    
    % Choosing pooling method and compiling video-level feature vector
    % of the video sequence
    if(strcmp(Constants.PoolMethod, 'max'))
        VideoLevelFeatures = max(FrameLevelFeatures,[],1); % max pooling
    elseif(strcmp(Constants.PoolMethod, 'min'))
        VideoLevelFeatures = min(FrameLevelFeatures,[],1); % min pooling
    elseif(strcmp(Constants.PoolMethod, 'median'))
        VideoLevelFeatures = median(FrameLevelFeatures,1); % median pooling
    elseif(strcmp(Constants.PoolMethod, 'avg'))
        VideoLevelFeatures = mean(FrameLevelFeatures,1);   % average pooling
    else
        error('Unknown base architecture');
    end
end

