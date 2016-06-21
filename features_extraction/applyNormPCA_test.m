
%%%%%%% Uses the previously computed parameters on training data for 
%%%%%%% normalizing the features and applying PCA
addpath('../Utils');

%% Parameters


%%%%%%% MSVD
images_path = '/media/HDD_2TB/DATASETS/MSVD';
features_csv = {{'train_C3Dfc8.csv', 'test_C3Dfc8.csv', 'val_C3Dfc8.csv'}, {'train_ImageNet.csv', 'test_ImageNet.csv', 'val_ImageNet.csv'}, {'train_ImageNetFV.csv', 'test_ImageNetFV.csv', 'val_ImageNetFV.csv'}, {'train_Places.csv', 'test_Places.csv', 'val_Places.csv'}}; % the first element in each list must always be the training features
pca_data_folder = 'PCA_data';

nFinalFeatures = 400; % number of desired features after PCA dimensionality reduction

%% Process each feature type separately
nTypes = length(features_csv);
for f = 1:nTypes
    
    params = load([images_path '/' pca_data_folder '/' features_csv{f}{1} 'parameters_PCA.mat']);
    training_parameters = params.training_parameters;
    
    n_splits = length(features_csv{f});
    for s = 1:n_splits
        disp(['Applying PCA on features ' features_csv{f}{s}]);
        featcsv = fileread([images_path '/' features_csv{f}{s}]);
        featcsv = regexp(featcsv, '\n', 'split');
        if(isempty(featcsv{end}))
            featcsv = {featcsv{1:end-1}};
        end
        nImages = length(featcsv);

        % Prepare output file
        name = regexp(features_csv{f}{s}, '\.', 'split');
        name = [name{1} '_PCA_dim' num2str(nFinalFeatures) '.csv'];
        file = fopen([images_path '/' name], 'w');

    	feat = cellfun(@str2num, regexp(featcsv{1}, ',', 'split'));
    	features_ImageNet = zeros(nImages, size(feat,2), 'single');
        disp('Reading features...');
    	for i = 1:nImages
            feat = cellfun(@str2num, regexp(featcsv{i}, ',', 'split')); 
            features_ImageNet(i,:) = single(feat);
            if(mod(i, 200) == 0 || i == nImages)
                disp(['Recovered features of ' num2str(i) '/' num2str(nImages) ' images.']);
            end
    	end

	disp('Computing PCA...');
        % Center
        features_ImageNet = features_ImageNet - repmat(training_parameters.center_ImageNet, size(features_ImageNet,1), 1);
        % Apply PCA
        features_ImageNet = features_ImageNet * training_parameters.pca_coeff_ImageNet(:, 1:nFinalFeatures);
    
	for i = 1:nImages
            % Save result
            fprintf(file, [strjoin(strread(num2str(features_ImageNet(i,:)),'%s')', ',') '\n']);
        
            if(mod(i, 200) == 0 || i == nImages)
            	disp(['Stored PCA result of ' num2str(i) '/' num2str(nImages) ' images.']);
            end
   	end

        fclose(file);
    end
    clear features_ImageNet;
    
end

disp('Done');
exit;
