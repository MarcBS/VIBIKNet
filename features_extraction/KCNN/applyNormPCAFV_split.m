
%%%%%%% Prepares the fisher vector features matrices for each train/val/test split
addpath('../../Utils;../../yael');


% %%%%%%% Flickr8k
% images_path = '/media/HDD_2TB/DATASETS';
% folders = {'Flickr8k/Flicker8k_Dataset'};
% 
% features_folders = {'GoogleNet_ImageNet'};
% kcnn_data_folders = {''};
% 
% lists_path = {[images_path '/Flickr8k/text']};
% % train, val test
% list_names = {'train', 'val', 'test'};
% lists = {{'Flickr_8k.trainImages.txt', 'Flickr_8k.devImages.txt', 'Flickr_8k.testImages.txt'}};


%%%%%% Flickr30k
% images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr30k/flickr30k-images'};
%
%features_folders = {'GoogleNet_ImageNet_ACL_16_task1'};
%kcnn_data_folders = {'KCNN_ACL_16_task1'};
%
%% lists_path = {[images_path '/Flickr30k/flickr30k-images']};
%lists_path = {[images_path '/Flickr30k/ACL_16_task1/split']};
%% train, val test
%list_names = {'train', 'val', 'test'};
%% lists = {{'train_list.txt', 'val_list.txt', 'test_list.txt'}}; % list of training videos
%lists = {{'train_images.txt', 'val_images.txt', 'test_images.txt'}}; % list of training videos

%%%%%%% MSVD
%images_path = '/media/HDD_2TB/DATASETS/MSVD'; % root of the videos database
%folders = {'Images'}; % folders where the images are stored
%format = '.jpg';
%
%lists_path = {images_path};
%% train, val test
%list_names = {'train', 'val', 'test'};
%lists = {{'train_list.txt', 'val_list.txt', 'test_list.txt'}}; % list of training videos

%%%%%% VQA
images_path = '/media/HDD_2TB/DATASETS/VQA';
folders = {'Images/mscoco/train2014', 'Images/mscoco/val2014', 'Images/mscoco/test2015'};
features_folders = {'GoogleNet_ImageNet'};
kcnn_data_folders = {'Features_KCNN'};
lists_path = {[images_path '/Images/mscoco']};
% train, val test
list_names = {'train', 'val', 'test'};
lists = {{'train_list.txt'}, {'val_list.txt'}, {'test_list.txt'}};


% Images subsampling applied for reducing the need of computational resources when extracting CNN features
subsampling = 1; % we can only process 1 out of 10 images (only valid for videos)

list_type = 'images'; % 'images' or 'videos'
nFisherVectors = 128*128; % input PCA features x number of GMMs
%nFisherVectors = 180*180;
%nFisherVectors = 64*64;
nFinalFeatures = 1024; % number of desired features after PCA dimensionality reduction

%% Process each folder separately
nFolders = length(folders);
for f = 1:nFolders
    %% Process each list
    nLists = length(list_names);
    for l = 1:nLists

        if(length(lists{f}) >= l) % higher indentation
        

        if(strcmp(list_type, 'videos'))
            disp(['Processing folder ' images_path '-' list_names{l}]);
        else
            disp(['Processing folder ' folders{f} '-' list_names{l}]);
        end
        
        %% Load previously computed parameters
        if(strcmp(list_type ,'images'))
            % Flickr
            %params = load([images_path '/' folders{f} '/' kcnn_data_folders{f} '/parameters_PCA_FV.mat']);
            % VQA
            params = load([images_path '/' folders{1} '/' kcnn_data_folders{1} '/parameters_PCA_FV.mat']);
        else
            params = load([images_path '/' kcnn_data_folders{f} '/parameters_PCA_FV.mat']);
        end
        pca_fv_parameters = params.pca_fv_parameters;
        
        %% Load list of training images or videos
        if(strcmp(list_type, 'videos'))
            % Load list of videos
            list_videos = fileread([lists_path{f} '/' lists{f}{l}]);
            list_videos = regexp(list_videos, '\n', 'split');
            if(isempty(list_videos{end}))
                list_videos = {list_videos{1:end-1}};
            end
            nVideos = length(list_videos);

            % List all images in all videos from this list
            list_images = cell(1, 1000000);
            list_images_pre = cell(1, 1000000);
	    matching_list = zeros(1, 1000000);
	    num_per_folder = [];
            nImages = 0;
	    nImages_complete = 0;
            for v = 1:nVideos
                path_list = [images_path '/' folders{f} '/' list_videos{v} '/'];
                
		% Complete list of images
		if(subsampling > 1)
		    list_complete_images = dir([path_list '*' format]);
		    list_complete_images = list_complete_images(arrayfun(@(x) x.name(1) ~= '.', list_complete_images));
    		    list_complete_images = {list_complete_images(:).name};
    		    num = length(list_complete_images);
    		    selected_images = round(linspace(1, num, num/subsampling));
		    % Find matching subsampled images for assigning the closest feature vector to the non processed images 
		    for i_comp = 1:num
		       	[~, closest_frame] = min(abs(selected_images-i_comp));
			matching_list(i_comp+nImages_complete) = closest_frame + nImages;
		    end
		    nImages_complete = nImages_complete + num;
		    num_per_folder(v) = num;
		end

		% List of extracted features
            	list = dir([path_list '/' features_folders{f} '/*_ImageNet_FV.mat']);

		list = list(arrayfun(@(x) x.name(1) ~= '.', list));
                list = {list(:).name};
                num = length(list);
		if(subsampling == 1)
		    num_per_folder(v) = num;
		end
                list_images(nImages+1:nImages+num) = list;
                list_images_pre(nImages+1:nImages+num) = repmat({list_videos{v}}, 1, num);
                nImages = nImages+num;
            end
            list_images = {list_images{1:nImages}};
            list_images_pre = {list_images_pre{1:nImages}};
	    matching_list = matching_list(1:nImages_complete);
        else
            % Load list of images
            % Flickr
            %list_images = fileread([lists_path{f} '/' lists{f}{l}]);
            % VQA
            list_images = fileread([lists_path{1} '/' lists{f}{l}]);
            list_images = regexp(list_images, '\n', 'split');
            if(isempty(list_images{end}))
                list_images = {list_images{1:end-1}};
            end
            nImages = length(list_images);
        end
        

        %% Load each image and store ImageNet features
        features = zeros(nImages, nFisherVectors);

        disp('Recovering ImageNet features...');
        for i = 1:nImages
            if(strcmp(list_type ,'images'))
                % Flickr
                %feat = load([images_path '/' folders{f} '/' features_folders{f} '/' list_images{i} '_ImageNet_FV.mat']);
                % VQA
                feat = load([images_path '/' folders{f} '/' features_folders{1} '/' list_images{i} '_ImageNet_FV.mat']);
            else
%                 feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i} '_ImageNet_FV.mat']);
                feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/' features_folders{f} '/' list_images{i}]);
            end
            
            features(i,:) = feat.fv_ImageNet;

            if(mod(i, 250)==0 || i == nImages)
                disp(['Recovered from ' num2str(i) '/' num2str(nImages) ' images.']);
            end
        end
        
%        %% Normalize 0-1
%       [features] = normalize(features, pca_fv_parameters.min_norm_ImageNet, pca_fv_parameters.max_norm_ImageNet);

%       %% Normalize L2
%        % power "normalization"
%        features = features';
%        features = sign(features) .* sqrt(abs(features));
%        % L2 normalization (may introduce NaN vectors)
%        [features] = yael_fvecs_normalize (features);
%        % replace NaN vectors with a large value that is far from everything else
%        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
%        % many vectors.
%        features(find(isnan(features))) = 123;
%        features = features';

        %% L2 sample normalization
        mag_features = single(sqrt(sum(features.^2,2)));
        features = features ./ repmat(mag_features, 1, size(features,2));

        %% Center data
        features = features - repmat(pca_fv_parameters.center_ImageNet, size(features,1), 1);

        %% Apply PCA
        features = features * pca_fv_parameters.pca_coeff_ImageNet(:, 1:nFinalFeatures);

        %% L2 sample normalization
        mag_features = single(sqrt(sum(features.^2,2)));
        features = features ./ repmat(mag_features, 1, size(features,2));

	%% Get final matrix by matching closes images
	if(subsampling > 1)
	    tmp = features;
	    features = zeros(nImages_complete, nFisherVectors);
	    features = tmp(matching_list, :);
	end

        %% Save final features matrix
        if(strcmp(list_type ,'images'))
            % Flickr
            %save([images_path '/' folders{f} '/' kcnn_data_folders{f} '/' list_names{l} '_ImageNet_FisherVectors.mat'], 'features');
            % VQA
             save([images_path '/' folders{1} '/' kcnn_data_folders{1} '/' list_names{f} '_ImageNet_FisherVectors.mat'], 'features');
        else
	    nImages_offset = 0;
	    tmp = features;
	    features = cell(1, nVideos);
	    for v = 1:nVideos
		features{v} = tmp(nImages_offset+1:nImages_offset+num_per_folder(v), :);
		nImages_offset = nImages_offset + num_per_folder(v);
	    end
            save([images_path '/' kcnn_data_folders{f} '/' list_names{l} '_ImageNet_FisherVectors.mat'], 'features');
        end


        
%         %% Load each image and store Places features
%         features = zeros(nImages, nFisherVectors);
% 
%         disp('Recovering Places features...');
%         for i = 1:nImages
%             if(strcmp(list_type ,'images'))
%                 feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat']);
%             else
%                 feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places_FV.mat']);
%             end
%             
%             features(i,:) = feat.fv_Places;
% 
%             if(mod(i, 250)==0 || i == nImages)
%                 disp(['Recovered from ' num2str(i) '/' num2str(nImages) ' images.']);
%             end
%         end
%         
% %        %% Normalize 0-1
% %        [features] = normalize(features, pca_fv_parameters.min_norm_Places, pca_fv_parameters.max_norm_Places);
% 
% %       %% Normalize L2
% %        % power "normalization"
% %        features = features';
% %        features = sign(features) .* sqrt(abs(features));
% %        % L2 normalization (may introduce NaN vectors)
% %        [features] = yael_fvecs_normalize (features);
% %        % replace NaN vectors with a large value that is far from everything else
% %        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
% %        % many vectors.
% %        features(find(isnan(features))) = 123;
% %        features = features';
% 
%         %% L2 sample normalization
%         mag_features = single(sqrt(sum(features.^2,2)));
%         features = features ./ repmat(mag_features, 1, size(features,2));
% 
%         %% Center data
%         features = features - repmat(pca_fv_parameters.center_Places, size(features,1), 1);
% 
%         %% Apply PCA
%         features = features * pca_fv_parameters.pca_coeff_Places(:, 1:nFinalFeatures);
% 
%         %% L2 sample normalization
%         mag_features = single(sqrt(sum(features.^2,2)));
%         features = features ./ repmat(mag_features, 1, size(features,2));
% 
%         %% Save final features matrix
%         if(strcmp(list_type ,'images'))
%             save([images_path '/' folders{f} '/' list_names{l} '_Places_FisherVectors.mat'], 'features');
%         else
%             save([images_path '/' list_names{l} '_Places_FisherVectors.mat'], 'features');
%         end

        end % higher indentation        

    end
end

disp('Done');
exit;
