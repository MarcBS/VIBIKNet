
%%%%%%% Applies learned GMM for extracting Fisher Vectors
% addpath('../../inria_fisher_v1');
addpath('../../yael');

%% Parameters
% images_path = '.';
% folders = {'test_images'};
% format = {'.jpg'};

%%%%%%% Flickr8k
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr8k/Flicker8k_Dataset'};
%format = {'.jpg'};
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
folders = {'Images/mscoco/train2014', 'Images/mscoco/val2014', 'Images/mscoco/test2015'};
%folders = {'Images/mscoco/test2015'};
format = {'.jpg', '.jpg', '.jpg'};
features_folders = {'GoogleNet_ImageNet', 'GoogleNet_ImageNet', 'GoogleNet_ImageNet'};
kcnn_data_folders = 'Images/mscoco/train2014/Features_KCNN';


list_type = 'images'; % 'images' or 'videos'


%% Process each folder separately
nFolders = length(folders);
for f = 1:nFolders
    
    %% Load training GMM parameters
    if(strcmp(list_type ,'images'))
        % Flickr version
        % load([images_path '/' folders{f} '/' kcnn_data_folders{f} '/parameters_GMM.mat']);
        % VQA (MSCOCO) version
        load([images_path '/' kcnn_data_folders '/parameters_GMM.mat']);
    else
        load([images_path '/' kcnn_data_folders{f} '/parameters_GMM.mat']);
    end
    
    %% Load list of training images or videos
    if(strcmp(list_type, 'videos'))
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
            list = dir([path_list '/' features_folders{f} '/*_ImageNet_PCA.mat']);

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
        % Load list of images
        list_images = dir([images_path '/' folders{f} '/*' format{f}]);
        list_images = list_images(arrayfun(@(x) x.name(1) ~= '.', list_images));
        list_images = {list_images(:).name};
        nImages = length(list_images);
    end
    
    %% Apply Fisher Vectors computation on ImageNet features
    
    %%% Load features from all training images
    disp('Computing Fisher Vectors on ImageNet features for each image...');
    
    for i = 1:nImages
        if(strcmp(list_type ,'images'))
            feat = load([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_PCA.mat']);
        else
%             feat = load([images_path '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i} '_ImageNet_PCA.mat']);
            feat = load([images_path '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i}]);
        end

        % L2 sample normalization
    	mag_features_ImageNet = single(sqrt(sum(feat.features_ImageNet.^2,2)));
    	feat.features_ImageNet = feat.features_ImageNet ./ repmat(mag_features_ImageNet, 1, size(feat.features_ImageNet,2));

        fv_ImageNet = yael_fisher(feat.features_ImageNet', gmm_parameters.w_ImageNet, ...
                gmm_parameters.mu_ImageNet, gmm_parameters.sigma_ImageNet, 'nonorm');
        
        % Save result
        if(strcmp(list_type ,'images'))
            save([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_FV.mat'], 'fv_ImageNet');
        else
            im_name = regexp(list_images{i}, '_', 'split');
            im_name = strjoin({im_name{1:end-2}}, '_');
            save([images_path '/' list_images_pre{i} '/' features_folders{f} '/' im_name '_ImageNet_FV.mat'], 'fv_ImageNet');
        end
        
        if(mod(i, 200) == 0 || i == nImages)
            disp(['Applied FVs on ' num2str(i) '/' num2str(nImages) ' images.']);
        end
    end
    
    
%     %% Apply Fisher Vectors computation on Places features
%     
%     %%% Load features from all training images
%     disp('Computing Fisher Vectors on Places features for each image...');
%     
%     for i = 1:nImages
%         if(strcmp(list_type ,'images'))
%             feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_PCA.mat']);
%         else
%             feat = load([images_path '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places_PCA.mat']);
%         end
%         
%         % L2 sample normalization
%         mag_features_Places = single(sqrt(sum(feat.features_Places.^2,2)));
%         feat.features_Places = feat.features_Places ./ repmat(mag_features_Places, 1, size(feat.features_Places,2));
% 
%         fv_Places = yael_fisher(feat.features_Places', gmm_parameters.w_Places, ...
%                 gmm_parameters.mu_Places, gmm_parameters.sigma_Places, 'nonorm');
%         
%         % Save result
%         if(strcmp(list_type ,'images'))
%             save([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat'], 'fv_Places');
%         else
%             save([images_path '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat'], 'fv_Places');
%         end
%         
%         if(mod(i, 200) == 0 || i == nImages)
%             disp(['Applied FVs on ' num2str(i) '/' num2str(nImages) ' images.']);
%         end
%     end
    
end

disp('Done');
exit;

