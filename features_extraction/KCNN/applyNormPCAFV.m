
%%%%%%% Normalizes the FisherVector features and applies PCA
addpath('../../Utils;../../yael');

%% Parameters
% images_path = '.';
% folders = {'test_images'};
% train_list = 'test_images/train_list.txt';

% %%%%%%% Flickr8k
% images_path = '/media/HDD_2TB/DATASETS';
% folders = {'Flickr8k/Flicker8k_Dataset'};
% features_folders = {'GoogleNet_ImageNet'};
% kcnn_data_folders = {''};
% train_list = [images_path '/Flickr8k/text/Flickr_8k.trainImages.txt'];


%%%%%% Flickr30k
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr30k/flickr30k-images'};
%features_folders = {'GoogleNet_ImageNet_ACL_16_task1'};
%kcnn_data_folders = {'KCNN_ACL_16_task1'};
%% train_list = [images_path '/Flickr30k/flickr30k-images/train_list.txt'];
%train_list = [images_path '/Flickr30k/ACL_16_task1/split/train_images.txt'];


%%%%%%% MSVD
%images_path = '/media/HDD_2TB/DATASETS/MSVD'; % root of the videos database
%folders = {'Images'}; % folders where the images are stored
%% format = '.jpg';
%train_list = [images_path '/train_list.txt']; % list of training videos


%%%%%% VQA
images_path = '/media/HDD_2TB/DATASETS/VQA';
folders = {'Images/mscoco/train2014'};
features_folders = {'GoogleNet_ImageNet'};
kcnn_data_folders = {'Features_KCNN'};
train_list = [images_path '/Images/mscoco/train_list.txt'];



list_type = 'images'; % 'images' or 'videos'
nFeatures = {16384, 16384}; % number of features extracted from each network [GoogleNet-Imagenet, GoogleNet-Places]
%nFeatures = {32400, 32400};
%nFeatures = {4096, 4096};

%% Process each folder separately
nFolders = length(folders);
for f = 1:nFolders

    %% Load list of training images or videos
    if(strcmp(list_type, 'videos'))
        disp(['Applying PCA FV on folder ' images_path]);

        % Load list of videos
        list_videos = fileread(train_list);
        list_videos = regexp(list_videos, '\n', 'split');
        if(isempty(list_videos{end}))
            list_videos = {list_videos{1:end-1}};
        end
        nVideos = length(list_videos);

        % List all images in all training videos
        list_images = cell(1, 1000000);
        list_images_pre = cell(1, 1000000);
        nImages = 0;
        for v = 1:nVideos
            path_list = [images_path '/' folders{f} '/' list_videos{v} '/'];
            
%           list = dir([path_list '*' format]);
            list = dir([path_list '/' features_folders{f} '/*_ImageNet_FV.mat']);

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
        disp(['Applying PCA FV on folder ' folders{f}]);

        % Load list of images
        list_images = fileread(train_list);
        list_images = regexp(list_images, '\n', 'split');
        if(isempty(list_images{end}))
            list_images = {list_images{1:end-1}};
        end
        nImages = length(list_images);
    end
    
    
    %% Process ImageNet features

    %%% Load features from all training images
    disp('Recovering samples ImageNet FisherVectors features...');
    features_ImageNet = zeros(nImages, nFeatures{1}, 'single');
    pca_fv_parameters = struct();
    
    for i = 1:nImages
        if(strcmp(list_type ,'images'))
            feat = load([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_FV.mat']);
        else
%             feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i} '_ImageNet_FV.mat']);
            feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i}]);
        end
        
        features_ImageNet(i, :) = single(feat.fv_ImageNet);
        
        if(mod(i, 200)==0 || i == nImages)
            disp(['Recovered from ' num2str(i) '/' num2str(nImages) ' images.']);
        end
    end

    %%% Normalize 0-1
%    disp('Applying features normalization...');
%    [features_ImageNet, pca_fv_parameters.min_norm_ImageNet, pca_fv_parameters.max_norm_ImageNet] = normalize(features_ImageNet);

%    %%% Normalize L2
%    % power "normalization"
%    features_ImageNet = features_ImageNet';
%    features_ImageNet = sign(features_ImageNet) .* sqrt(abs(features_ImageNet));
%    % L2 normalization (may introduce NaN vectors)
%    [features_ImageNet] = yael_fvecs_normalize (features_ImageNet);
%    % replace NaN vectors with a large value that is far from everything else
%    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
%    % many vectors.
%    features_ImageNet(find(isnan(features_ImageNet))) = 123;
%    features_ImageNet = features_ImageNet';

    %%% L2 sample normalization
    mag_features_ImageNet = single(sqrt(sum(features_ImageNet.^2,2)));
    features_ImageNet = features_ImageNet ./ repmat(mag_features_ImageNet, 1, size(features_ImageNet,2));

    %%% Center data
    pca_fv_parameters.center_ImageNet = single(mean(features_ImageNet));
    features_ImageNet = features_ImageNet - repmat(pca_fv_parameters.center_ImageNet, size(features_ImageNet,1), 1);


    %%% Apply PCA
    disp('Applying PCA...');
    pca_fv_parameters.pca_coeff_ImageNet = pca(features_ImageNet);

    clear features_ImageNet;

    
%     %% Process Places features
% 
%     %%% Load features from all training images
%     disp('Recovering samples Places FisherVectors features...');
%     features_Places = zeros(nImages, nFeatures{2}, 'single');
%     
%     for i = 1:nImages
%         if(strcmp(list_type ,'images'))
%             feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat']);
%         else
%             feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat']);
%         end
%         
%         features_Places(i, :) = single(feat.fv_Places);
%         
%         if(mod(i, 200)==0 || i == nImages)
%             disp(['Recovered from ' num2str(i) '/' num2str(nImages) ' images.']);
%         end
%     end
% 
%     %%% Normalize 0-1
% %    disp('Applying features normalization...');
% %    [features_Places, pca_fv_parameters.min_norm_Places, pca_fv_parameters.max_norm_Places] = normalize(features_Places);
% 
% %    %%% Normalize L2
% %    % power "normalization"
% %    features_Places = features_Places';
% %    features_Places = sign(features_Places) .* sqrt(abs(features_Places));
% %    % L2 normalization (may introduce NaN vectors)
% %    [features_Places] = yael_fvecs_normalize (features_Places);
% %    % replace NaN vectors with a large value that is far from everything else
% %    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
% %    % many vectors.
% %    features_Places(find(isnan(features_Places))) = 123;
% %    features_Places = features_Places';
% 
%     %%% L2 sample normalization
%     mag_features_Places = single(sqrt(sum(features_Places.^2,2)));
%     features_Places = features_Places ./ repmat(mag_features_Places, 1, size(features_Places,2));
% 
%     %%% Center data
%     pca_fv_parameters.center_Places = single(mean(features_Places));
%     features_Places = features_Places - repmat(pca_fv_parameters.center_Places, size(features_Places,1), 1);
% 
%     %%% Apply PCA
%     disp('Applying PCA...');
%     pca_fv_parameters.pca_coeff_Places = pca(features_Places);
% 
%     clear features_Places;
    
    %% Save parameters
    disp('Saving PCA FV result...');
    if(strcmp(list_type ,'images'))
        save([images_path '/' folders{f} '/' kcnn_data_folders{f} '/parameters_PCA_FV.mat'], 'pca_fv_parameters', '-v7.3');
    else
        save([images_path '/' kcnn_data_folders{f} '/parameters_PCA_FV.mat'], 'pca_fv_parameters', '-v7.3');
    end
end

disp('Done');
exit;
