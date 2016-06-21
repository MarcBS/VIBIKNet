
%%%%%%% Normalizes the features and applies PCA
addpath('../Utils');

%% Parameters

%%%%%%% MSVD
images_path = '/media/HDD_2TB/DATASETS/MSVD'; % root of the videos database
features_csv = {'train_C3Dfc8.csv', 'train_ImageNet.csv', 'train_ImageNetFV.csv', 'train_Places.csv'};
nFeatures = {487, 1024, 1024, 1024};
pca_data_folder = 'PCA_data';
maxSamples = 800000;


%% Process each features separately
nFeaturesCSV = length(features_csv);
for f = 1:nFeaturesCSV

    	disp(['Applying PCA on features ' features_csv{f}]);

	% Load list of images
    	list_images = fileread([images_path '/' features_csv{f}]);
    	list_images = regexp(list_images, '\n', 'split');
    	if(isempty(list_images{end}))
            list_images = {list_images{1:end-1}};
    	end
    	nImages = length(list_images);
    
    %% Randomly select samples for each of the training images 
    totalSamples = nImages;
    maxSamples = min([totalSamples maxSamples]);
    samplesPerImage = max([1 floor(maxSamples/nImages)]);

    selected_images = 1:nImages;
    maxSamples = length(selected_images);

    
    %% Process ImageNet features

    %%% Load features from all training images
    disp('Recovering samples ImageNet features...');
    features_ImageNet = zeros(maxSamples, nFeatures{f}, 'single');
    training_parameters = struct();
    
    offset = 0;
    nSelected = length(selected_images);
    for i = selected_images
        feat = cellfun(@str2num, regexp(list_images{i}, ',', 'split'));
        features_ImageNet(offset+1, :) = single(feat);
        offset = offset+1;
        
        if(mod(i, 200)==0 || i == nSelected)
            disp(['Recovered from ' num2str(i) '/' num2str(nSelected) ' images.']);
        end
    end

%    %%% Normalize
%    disp('Applying features normalization...');
%    [features_ImageNet, training_parameters.min_norm_ImageNet, training_parameters.max_norm_ImageNet] = normalize(features_ImageNet);
    %%% Center data
    training_parameters.center_ImageNet = single(mean(features_ImageNet));
    features_ImageNet = features_ImageNet - repmat(training_parameters.center_ImageNet, size(features_ImageNet,1), 1); 

    %%% Apply PCA
    disp('Applying PCA...');
    training_parameters.pca_coeff_ImageNet = princomp(features_ImageNet);

    clear features_ImageNet;

%     %% Process Places features
%     disp('Recovering samples Places features...');
%     features_Places = zeros(maxSamples, nFeatures{2}, 'single');
%     
%     offset = 0;
%     nSelected = length(selected_images);
%     for i = selected_images
% 	if(strcmp(list_type ,'images'))
%             feat = load([images_path '/' folders{f} '/GoogleNet_Places/' list_images{i} '_Places.mat']);
% 	else
% 	    feat = load([images_path '/' folders{f} '/' list_images_pre{i} '/GoogleNet_Places/' list_images{i} '_Places.mat'])
% 	end
%         randselect = randsample(1:size(feat.features_Places,1), samplesPerImage);
%         features_Places(offset+1:offset+samplesPerImage, :) = single(feat.features_Places(randselect,:));
%         offset = offset+samplesPerImage;
%         
%         if(mod(i, 200)==0 || i == nSelected)
%             disp(['Recovered from ' num2str(i) '/' num2str(nSelected) ' images.']);
%         end
%     end
% 
% %    %%% Normalize
% %    disp('Applying features normalization...');
% %    [features_Places, training_parameters.min_norm_Places, training_parameters.max_norm_Places] = normalize(features_Places);
%     %%% Center data
%     training_parameters.center_Places = single(mean(features_Places));
%     features_Places = features_Places - repmat(training_parameters.center_Places, size(features_Places,1), 1);
% 
%     %%% Apply PCA
%     disp('Applying PCA...');
%     training_parameters.pca_coeff_Places = princomp(features_Places);
% 
%     clear features_Places;

    %%% Save parameters
    disp('Saving result...');
	if(~exist([images_path '/' pca_data_folder]))
            mkdir([images_path '/' pca_data_folder])
    	end
    	save([images_path '/' pca_data_folder '/' features_csv{f} 'parameters_PCA.mat'], 'training_parameters');
end

disp('Done');
exit;
