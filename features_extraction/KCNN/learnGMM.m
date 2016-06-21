
%%%%%%% Learns Gaussian Mixture Models (GMM) on the PCA features
addpath('../../yael');

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
nFeaturesPCA = 128;
%nFeaturesPCA = 64;
%nFeaturesPCA = 180;

maxSamples = 10000000; % total ~6000 samples x 800 bb = 4.800.000 ~2.3GB. 15.000.000 ~= 7.2GB
samplesPerImage = 240; %800; % number of bounding boxes processed per image [nRotations * nObjectProposals]

nGMMs = 128;
%nGMMs = 16;
%nGMMs = 32;
%nGMMs = 180;
%nGMMs = 64;

%% Process each folder separately
nFolders = length(folders);
for f = 1:nFolders
    
    gmm_parameters = struct();
    
    %% Load list of training images or videos
    if(strcmp(list_type, 'videos'))
        disp(['Learning GMM on folder ' images_path]);

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
        disp(['Learning GMM on folder ' folders{f}]);

        % Load list of images
        list_images = fileread(train_list);
        list_images = regexp(list_images, '\n', 'split');
        if(isempty(list_images{end}))
            list_images = {list_images{1:end-1}};
        end
        nImages = length(list_images);
    end
    
    %% Randomly select samples for each of the training images 
    totalSamples = samplesPerImage*nImages;
    maxSamples = min([totalSamples maxSamples]);
    samplesPerImage = max([1 floor(maxSamples/nImages)]);

    if(strcmp(list_type ,'videos'))
        if(samplesPerImage < 15)
            samplesPerImage = 15;
            selected_images = randsample(1:nImages, floor(maxSamples/samplesPerImage));
        else
            selected_images = 1:nImages;
        end
        disp(['Picking ' num2str(samplesPerImage) ' random feature vectors per image from ' num2str(length(selected_images)) '/' num2str(nImages) ' images...'])
    else
        selected_images = 1:nImages;
        disp(['Picking ' num2str(samplesPerImage) ' random feature vectors per image...'])
    end

    maxSamples = samplesPerImage*length(selected_images);
    
    %% GMMs computation on ImageNet features
    
    features_ImageNet = zeros(maxSamples, nFeaturesPCA, 'single');
    
    %%% Load features from all training images
    disp('Recovering ImageNet PCA features for each image...');

    offset = 0;
    nSelected = length(selected_images);
    for i = selected_images
        if(strcmp(list_type ,'images'))
            feat = load([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_PCA.mat']);
        else
%             feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i} '_ImageNet_PCA.mat']);
            feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i}]);
        end
        randselect = randsample(1:size(feat.features_ImageNet,1), samplesPerImage);
        features_ImageNet(offset+1:offset+samplesPerImage, :) = single(feat.features_ImageNet(randselect,:));
        offset = offset+samplesPerImage;
        
        if(mod(i, 200)==0 || i == nSelected)
            disp(['Recovered from ' num2str(i) '/' num2str(nSelected) ' images.']);
        end
    end

    %% L2 sample normalization
    mag_features_ImageNet = single(sqrt(sum(features_ImageNet.^2,2)));
    features_ImageNet = features_ImageNet ./ repmat(mag_features_ImageNet, 1, size(features_ImageNet,2));
    
    %% Apply Gaussian Mixture Model (GMM)
    disp('Computing GMM on ImageNet features...');
    [gmm_parameters.w_ImageNet, gmm_parameters.mu_ImageNet, gmm_parameters.sigma_ImageNet] = yael_gmm(features_ImageNet', nGMMs);
    clear features_ImageNet;
    
    
%     %% GMMs computation on Places features
%     
%     features_Places = zeros(maxSamples, nFeaturesPCA, 'single');
%     
%     %%% Load features from all training images
%     disp('Recovering Places PCA features for each image...');
%     
%     offset = 0;
%     nSelected = length(selected_images);
%     for i = selected_images
%         if(strcmp(list_type ,'images'))
%             feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_PCA.mat']);
%         else
%             feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places_PCA.mat']);
%         end
%         randselect = randsample(1:size(feat.features_Places,1), samplesPerImage);
%         features_Places(offset+1:offset+samplesPerImage, :) = single(feat.features_Places(randselect,:));
%         offset = offset+samplesPerImage;
%         
%         if(mod(i, 200)==0 || i == nSelected)
%             disp(['Recovered from ' num2str(i) '/' num2str(nSelected) ' images.']);
%         end
%     end
%     
%     %% L2 sample normalization
%     mag_features_Places = single(sqrt(sum(features_Places.^2,2)));
%     features_Places = features_Places ./ repmat(mag_features_Places, 1, size(features_Places,2));
%     
%     %% Apply Gaussian Mixture Model (GMM)
%     disp('Computing GMM on Places features...');
%     [gmm_parameters.w_Places, gmm_parameters.mu_Places, gmm_parameters.sigma_Places] = yael_gmm(features_Places', nGMMs);
%     clear features_Places;
    
    
    %% Save result
    disp('Saving GMM parameters.');
    if(strcmp(list_type ,'images'))
        save([images_path '/' folders{f} '/' kcnn_data_folders{f} '/parameters_GMM.mat'], 'gmm_parameters');
    else
        save([images_path '/' kcnn_data_folders{f} '/parameters_GMM.mat'], 'gmm_parameters');
    end
end

disp('Done');
exit;
