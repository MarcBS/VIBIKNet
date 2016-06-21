
%% Example Data
data_path = '/media/HDD_2TB/DATASETS';
folders = {'COCO/test2014', 'COCO/val2014', 'COCO/train2014', 'Flickr8k/Flicker8k_Dataset', 'Flickr30k/flickr30k-images'};
formats = {};


%%%%%%
%
% Conv_v1 -> Aided-Hard Attention (Top 1)
% Conv_v2 -> Aided-Soft Attention (max Top 5)
%
%%%%%%


%% Add folders from MSRVD too
%msrvd = 'MSRVD/Images';
%folders_msrvd = dir([data_path '/' msrvd '/*']);
%folders_msrvd = folders_msrvd(arrayfun(@(x) x.name(1) ~= '.' && isdir([data_path '/' msrvd '/' x.name]), folders_msrvd));
%nNames = length(folders_msrvd);
%for n = 1:nNames
%	names{n} = [msrvd '/' folders_msrvd(n).name];
%end
%folders = {folders{:}, names{:}};
%nFolders = length(folders);
%for f = 1:nFolders
%	formats{f} = '.jpg';
%end


%%%%%% VQA
folders = {'VQA/Images/mscoco/train2014', 'VQA/Images/mscoco/val2014', 'VQA/Images/mscoco/test2015'};
formats = {'.jpg', '.jpg', '.jpg'};
store_paths = {[data_path '/VQA/Features/mscoco/train2014'], [data_path '/VQA/Features/mscoco/val2014'], [data_path '/VQA/Features/mscoco/test2015']};


%folders = {'Flickr8k/Flicker8k_Dataset'};
%formats = {'.jpg'};


%% CNN Parameters
CNN_params.caffe_path = '/home/lifelogging/code/caffe/';
CNN_params.use_gpu = 1;
CNN_params.gpu_id = 0;

%%% GoogleNet - ImageNet parameters
CNN_params.batch_size = 100; % Depending on the deploy net structure!!
CNN_params.model_def_file = '/media/HDD_2TB/CNN_MODELS/GoogleNet/deploy_conv_features_extraction.prototxt';
CNN_params.model_file = '/media/HDD_2TB/CNN_MODELS/GoogleNet/bvlc_googlenet.caffemodel';
CNN_params.input_size = 224;
CNN_params.image_mean = zeros(256, 256, 3);
CNN_params.image_mean(:,:,1) = 104; CNN_params.image_mean(:,:,2) = 117; CNN_params.image_mean(:,:,3) = 123;
CNN_params.size_features = 1024;
features_name = 'GoogleNet_ImageNet_Conv';

disp(' ');
disp('Extracting ImageNet features.');
disp(' ');
nFolders = length(folders);
for f = 1:nFolders
	disp(['Extracting folder ' folders{f} ' (' num2str(f) '/' num2str(nFolders) ')']);
        %% Run CNN and extract features
	[features_folders, coords] = extractCNNFeaturesConv(data_path, {folders{f}}, {formats{f}}, CNN_params);

        %% Store features
	for feat = 1:2
		features = features_folders{1}{feat};
		coordinates = coords{1}{feat};
		save([store_paths{f} '/' features_name '_v' num2str(feat) '.mat'], 'features');
		save([store_paths{f} '/' features_name '_v' num2str(feat) '_coordinates.mat'], 'coordinates');
		disp(' ');
	end
end


disp('Done ImageNet Conv');
exit


%%% GoogleNet - Places parameters
CNN_params.batch_size = 10; % Depending on the deploy net structure!!
CNN_params.model_def_file = '/media/HDD_2TB/CNN_MODELS/Places_CNN/googlenet/deploy_conv_feature_extraction.prototxt';
CNN_params.model_file = '/media/HDD_2TB/CNN_MODELS/Places_CNN/googlenet/googlelet_places205_train_iter_2400000.caffemodel';
CNN_params.input_size = 224;
load(['/media/HDD_2TB/CNN_MODELS/Places_CNN/placesCNN/places205CNN_mean.mat']); % image_mean
CNN_params.image_mean = permute(image_mean, [2 3 1]);
%CNN_params.size_features = 205;
CNN_params.size_features = 1024;
features_name = 'GoogleNet_Places_Conv';

disp(' ');
disp('Extracting Places205 features.');
disp(' ');
nFolders = length(folders);
for f = 1:nFolders
	disp(['Extracting folder ' folders{f} ' (' num2str(f) '/' num2str(nFolders) ')']);
	%% Run CNN and extract features
	[features_folders, coords] = extractCNNFeaturesConv(data_path, {folders{f}}, {formats{f}}, CNN_params);

	%% Store features
        for feat = 1:2
                features = features_folders{1}{feat};
                coordinates = coords{1}{feat};
                save([store_paths{f} '/' features_name '_v' num2str(feat) '.mat'], 'features');
		save([store_paths{f} '/' features_name '_v' num2str(feat) '_coordinates.mat'], 'coordinates');
                disp(' ');
        end
end

disp('Done');
exit;
