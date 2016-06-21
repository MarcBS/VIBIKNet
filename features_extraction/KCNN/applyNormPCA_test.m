
%%%%%%% Uses the previously computed parameters on training data for 
%%%%%%% normalizing the features and applying PCA
addpath('../../Utils');

%% Parameters
% images_path = '.';
% folders = {'test_images'};
% format = {'.jpg'};

%%%%%%% Flickr8k
% images_path = '/media/HDD_2TB/DATASETS';
% folders = {'Flickr8k/Flicker8k_Dataset'};
% format = {'.jpg'};
% features_folders = {'GoogleNet_ImageNet'};
% kcnn_data_folders = {''};

%%%%%% Flickr30k
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr30k/flickr30k-images'};
%format = {'.jpg'};
%features_folders = {'GoogleNet_ImageNet_ACL_16_task1'};
%kcnn_data_folders = {'KCNN_ACL_16_task1'};


%%%%%%% MSVD
%images_path = '/media/HDD_2TB/DATASETS/MSVD'; % root of the videos database
%folders = {};
%% format = '.jpg';
%
%msrvd = 'Images';
%folders_msrvd = dir([images_path '/' msrvd '/*']);
%folders_msrvd = folders_msrvd(arrayfun(@(x) x.name(1) ~= '.' && isdir([images_path '/' msrvd '/' x.name]), folders_msrvd));
%nNames = length(folders_msrvd);
%for n = 1:nNames
%       names{n} = [msrvd '/' folders_msrvd(n).name];
%end
%folders = {folders{:}, names{:}};
%folders = {folders};


%%%%%% VQA
images_path = '/media/HDD_2TB/DATASETS/VQA';
%folders = {'Images/mscoco/train2014', 'Images/mscoco/val2014', 'Images/mscoco/test2015'};
folders = {'Images/mscoco/test2015'};
format = {'.jpg', '.jpg', '.jpg'};
features_folders = {'GoogleNet_ImageNet'};
kcnn_data_folders = 'Images/mscoco/train2014/Features_KCNN';


list_type = 'images'; % 'images' or 'videos'
nFinalFeatures = 128; % number of desired features after PCA dimensionality reduction
%nFinalFeatures = 64;
%nFinalFeatures = 180;

%% Process each folder separately
nFolders = length(folders);
for f = 1:nFolders
    
    %% Load previously computed parameters
    if(strcmp(list_type ,'images'))
        % Flickr version
        % params = load([images_path '/' folders{f} '/' kcnn_data_folders{f} '/parameters_PCA.mat']);
        % VQA (MSCOCO) version
        params = load([images_path '/' kcnn_data_folders '/parameters_PCA.mat']);
    else
        params = load([images_path '/' kcnn_data_folders{f} '/parameters_PCA.mat']);
    end
    training_parameters = params.training_parameters;
    
    
    %% Load list of training images or videos
    if(strcmp(list_type, 'videos'))
        disp(['Applying PCA on folder ' images_path]);

        % Load list of videos
        list_videos = folders{f};
        nVideos = length(list_videos);

        % List all images in all videos
        list_images = cell(1, 2000000);
        list_images_pre = cell(1, 2000000);
        nImages = 0;
        for v = 1:nVideos
            path_list = [images_path '/' list_videos{v} '/'];
            
%           list = dir([path_list '*' format]);
            list = dir([path_list '/' features_folders{f} '/*_ImageNet.mat']);

	    list = list(arrayfun(@(x) x.name(1) ~= '.', list));
            list = {list(:).name};
            num = length(list);
            list_images(nImages+1:nImages+num) = list;
            list_images_pre(nImages+1:nImages+num) = repmat({list_videos{v}}, 1, num);
            nImages = nImages+num;
        end
        list_images = {list_images{1:nImages}};
        list_images_pre = {list_images_pre{1:nImages}};
    else
        disp(['Applying PCA on folder ' folders{f}]);

        % Load list of images
        list_images = dir([images_path '/' folders{f} '/*' format{f}]);
        list_images = list_images(arrayfun(@(x) x.name(1) ~= '.', list_images));
        list_images = {list_images(:).name};
        nImages = length(list_images);
    end
    
    
    %% Process ImageNet features

    %%% Load features from all training images
    disp('Applying PCA on ImageNet features for each image...');
    
    for i = 1:nImages
        if(strcmp(list_type ,'images'))
            feat = load([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet.mat']);
        else
%             feat = load([images_path '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i} '_ImageNet.mat']);
            feat = load([images_path '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i}]);
        end
        features_ImageNet = single(feat.features_ImageNet);
%        % Normalize
%        [features_ImageNet] = normalize(features_ImageNet, training_parameters.min_norm_ImageNet, training_parameters.max_norm_ImageNet);
        % Center
        features_ImageNet = features_ImageNet - repmat(training_parameters.center_ImageNet, size(features_ImageNet,1), 1);
        % Apply PCA
        features_ImageNet = features_ImageNet * training_parameters.pca_coeff_ImageNet(:, 1:nFinalFeatures);
%        features_ImageNet = features_ImageNet(:, 1:nFinalFeatures);

        % Save result
        if(strcmp(list_type ,'images'))
            save([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_PCA.mat'], 'features_ImageNet');
        else
	    im_name = regexp(list_images{i}, '_', 'split');
	    im_name = strjoin({im_name{1:end-1}}, '_'); 
            save([images_path '/' list_images_pre{i} '/' features_folders{f} '/' im_name '_ImageNet_PCA.mat'], 'features_ImageNet');
        end
        
        if(mod(i, 200) == 0 || i == nImages)
            disp(['Applied PCA on ' num2str(i) '/' num2str(nImages) ' images.']);
        end
    end
    clear features_ImageNet;
    
    
%     %% Process Places features
% 
%     %%% Load features from all training images
%     disp('Applying PCA on Places features for each image...');
%     
%     for i = 1:nImages
%         if(strcmp(list_type ,'images'))
%             feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places.mat']);
%         else
%             feat = load([images_path '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places.mat']);
%         end
%         features_Places = single(feat.features_Places);
% %        % Normalize
% %        [features_Places] = normalize(features_Places, training_parameters.min_norm_Places, training_parameters.max_norm_Places);
%         % Center
%         features_Places = features_Places - repmat(training_parameters.center_Places, size(features_Places,1), 1);
%         % Apply PCA
%         features_Places = features_Places * training_parameters.pca_coeff_Places(:, 1:nFinalFeatures);
% %        features_Places = features_Places(:, 1:nFinalFeatures);
% 
%         % Save result
%         if(strcmp(list_type ,'images'))
%             save([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_PCA.mat'], 'features_Places');
%         else
%  	      im_name = regexp(list_images{i}, '_', 'split');
%             im_name = strjoin({im_name{1:end-1}}, '_');            
% 	      save([images_path '/' list_images_pre{i} '/GoogleNet_ImageNet/' im_name '_ImageNet_PCA.mat'], 'features_ImageNet');
%         end
%             
%         if(mod(i, 200) == 0 || i == nImages)
%             disp(['Applied PCA on ' num2str(i) '/' num2str(nImages) ' images.']);
%         end
%     end
%     clear features_Places;

end

disp('Done');
exit;
