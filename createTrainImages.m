function [] = createTrainImages(TrainVideos, TrainMOS, Constants)
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end

    disp('Creating training images for transfer learning');
    numberOfVideos = Constants.numberOfTrainVideos;
    
    path = Constants.path;
    path2 = strrep(Constants.path, 'KoNViD_1k_videos', 'SortedFrames');
    
    if(exist('SortedFrames'))
        delete(strcat('SortedFrames', filesep, 'VeryGoodImages', filesep, '*.*'));
        delete(strcat('SortedFrames', filesep, 'GoodImages', filesep, '*.*'));
        delete(strcat('SortedFrames', filesep, 'MediocreImages', filesep, '*.*'));
        delete(strcat('SortedFrames', filesep, 'PoorImages', filesep, '*.*'));
        delete(strcat('SortedFrames', filesep, 'VeryPoorImages', filesep, '*.*'));
    else
        mkdir('SortedFrames');
        mkdir(strcat('SortedFrames', filesep, 'VeryGoodImages'));
        mkdir(strcat('SortedFrames', filesep, 'GoodImages'));
        mkdir(strcat('SortedFrames', filesep, 'MediocreImages'));
        mkdir(strcat('SortedFrames', filesep, 'PoorImages'));
        mkdir(strcat('SortedFrames', filesep, 'VeryPoorImages'));
    end
    
    i=1;
    for ind=1:numberOfVideos
        if(mod(ind,10)==0)
            disp(ind); 
        end
        v=VideoReader( char(strcat(path, filesep, TrainVideos{ind,1}, '.mp4')) );
    
        while hasFrame(v)
            frame = readFrame(v);
            if(rand>=0.8)     
                        
                if(TrainMOS(ind)<=1.8)
                    img = imresize(frame,[338 338]);
                    img = imcrop(img, [19.5 19.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryPoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>1.8 && TrainMOS(ind)<=2.6)
                    img = imresize(frame,[338 338]);
                    img = imcrop(img, [19.5 19.5 298 298]);
                    saveIm = strcat(path2, filesep, 'PoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>2.6 && TrainMOS(ind)<=3.4)
                    img = imresize(frame,[338 338]);
                    img = imcrop(img, [19.5 19.5 298 298]);
                    saveIm = strcat(path2, filesep, 'MediocreImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>3.4 && TrainMOS(ind)<=4.2)
                    img = imresize(frame,[338 338]);
                    img = imcrop(img, [19.5 19.5 298 298]);
                    saveIm = strcat(path2, filesep, 'GoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                else
                    img = imresize(frame,[338 338]);
                    img = imcrop(img, [19.5 19.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryGoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                end
            
                i=i+1;
            end
        end
    end

end

