
%% This script converts .mat files into .csv files
% path_mat = '/Volumes/KINGSTON M/Video_Description';
path_mat = '/media/HDD_2TB/DATASETS';
%mat_list = {{'MSRVD/test_GoogleNet_ImageNet', 'MSRVD/test_GoogleNet_Places'}, ...
%    {'MSRVD/train_GoogleNet_ImageNet', 'MSRVD/train_GoogleNet_Places'}, ...
%    {'MSRVD/val_GoogleNet_ImageNet', 'MSRVD/val_GoogleNet_Places'}, ...
%    {'COCO/test2014/test_GoogleNet_ImageNet', 'COCO/test2014/test_GoogleNet_Places'}, ...
%    {'COCO/train2014/train_GoogleNet_ImageNet', 'COCO/train2014/train_GoogleNet_Places'}, ...
%    {'COCO/val2014/val_GoogleNet_ImageNet', 'COCO/val2014/val_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/test_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/test_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/train_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/train_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/val_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/val_GoogleNet_Places'}, ...
%    {'Flickr30k/flickr30k-images/test_GoogleNet_ImageNet', 'Flickr30k/flickr30k-images/test_GoogleNet_Places'}, ...
%    {'Flickr30k/flickr30k-images/train_GoogleNet_ImageNet', 'Flickr30k/flickr30k-images/train_GoogleNet_Places'}, ...
%    {'Flickr30k/flickr30k-images/val_GoogleNet_ImageNet', 'Flickr30k/flickr30k-images/val_GoogleNet_Places'}}; % file pairs without '.mat' extension
%out_list = {'MSRVD/test', 'MSRVD/train', 'MSRVD/val', ...
%    'COCO/test2014/test', 'COCO/train2014/train', 'COCO/val2014/val', ...
%    'Flickr8k/Flicker8k_Dataset/test', 'Flickr8k/Flicker8k_Dataset/train', 'Flickr8k/Flicker8k_Dataset/val', ...
%    'Flickr30k/flickr30k-images/test', 'Flickr30k/flickr30k-images/train', 'Flickr30k/flickr30k-images/val'};

%mat_list = {{'Flickr8k/Flicker8k_Dataset/test_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/test_Places_FisherVectors'}, ...
%    {'Flickr8k/Flicker8k_Dataset/train_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/train_Places_FisherVectors'}, ...
%    {'Flickr8k/Flicker8k_Dataset/val_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/val_Places_FisherVectors'}};
%out_list = {'Flickr8k/Flicker8k_Dataset/test_KCNN', 'Flickr8k/Flicker8k_Dataset/train_KCNN', 'Flickr8k/Flicker8k_Dataset/val_KCNN'};

%mat_list = {{'Flickr8k/Flicker8k_Dataset/test_ImageNet_FisherVectors'}, ...
%    {'Flickr8k/Flicker8k_Dataset/train_ImageNet_FisherVectors'}, ...
%    {'Flickr8k/Flicker8k_Dataset/val_ImageNet_FisherVectors'}};
%out_list = {'Flickr8k/Flicker8k_Dataset/test_KCNN_ImageNet', 'Flickr8k/Flicker8k_Dataset/train_KCNN_ImageNet', 'Flickr8k/Flicker8k_Dataset/val_KCNN_ImageNet'};

%mat_list = {{'MSVD/test_GoogleNet_ImageNet', 'MSVD/test_GoogleNet_Places'}, ...
%    {'MSVD/train_GoogleNet_ImageNet', 'MSVD/train_GoogleNet_Places'}, ...
%    {'MSVD/val_GoogleNet_ImageNet', 'MSVD/val_GoogleNet_Places'}};
%out_list = {'MSVD/test', 'MSVD/train', 'MSVD/val'};

%mat_list = {{'Flickr30k/ACL_16_task1/test_GoogleNet_ImageNet', 'Flickr30k/ACL_16_task1/test_GoogleNet_Places'}, ...
%    {'Flickr30k/ACL_16_task1/train_GoogleNet_ImageNet', 'Flickr30k/ACL_16_task1/train_GoogleNet_Places'}, ...
%    {'Flickr30k/ACL_16_task1/val_GoogleNet_ImageNet', 'Flickr30k/ACL_16_task1/val_GoogleNet_Places'}};
%out_list = {'Flickr30k/ACL_16_task1/test_objects_places', 'Flickr30k/ACL_16_task1/train_objects_places', 'Flickr30k/ACL_16_task1/val_objects_places'};

%mat_list = {{'Flickr30k/flickr30k-images/KCNN_ACL_16_task1/test_ImageNet_FisherVectors', 'Flickr30k/ACL_16_task1/test_GoogleNet_Places'}, ...
%    {'Flickr30k/flickr30k-images/KCNN_ACL_16_task1/train_ImageNet_FisherVectors', 'Flickr30k/ACL_16_task1/train_GoogleNet_Places'}, ...
%    {'Flickr30k/flickr30k-images/KCNN_ACL_16_task1/val_ImageNet_FisherVectors', 'Flickr30k/ACL_16_task1/val_GoogleNet_Places'}};
%out_list = {'Flickr30k/ACL_16_task1/test_objectsKCNN_places', 'Flickr30k/ACL_16_task1/train_objectsKCNN_places', 'Flickr30k/ACL_16_task1/val_objectsKCNN_places'};

%mat_list = {{'MSVD/test_GoogleNet_ImageNet', 'MSVD/test_C3D_features'}, ...
%    {'MSVD/train_GoogleNet_ImageNet', 'MSVD/train_C3D_features'}, ...
%    {'MSVD/val_GoogleNet_ImageNet', 'MSVD/val_C3D_features'}};
%out_list = {'MSVD/test_C3D_ImageNet', 'MSVD/train_C3D_ImageNet', 'MSVD/val_C3D_ImageNet'};

%mat_list = {{'MSVD/test_GoogleNet_ImageNet'}, ...
%    {'MSVD/train_GoogleNet_ImageNet'}, ...
%    {'MSVD/val_GoogleNet_ImageNet'}};
%out_list = {'MSVD/test_ImageNet', 'MSVD/train_ImageNet', 'MSVD/val_ImageNet'};

%mat_list = {{'MSVD/test_GoogleNet_ImageNet', 'MSVD/test_C3D_features_fc8', 'MSVD/test_GoogleNet_Places'}, ...
%    {'MSVD/train_GoogleNet_ImageNet', 'MSVD/train_C3D_features_fc8', 'MSVD/train_GoogleNet_Places'}, ...
%    {'MSVD/val_GoogleNet_ImageNet', 'MSVD/val_C3D_features_fc8', 'MSVD/val_GoogleNet_Places'}};
%out_list = {'MSVD/test_C3D_fc8_ImageNet_Places', 'MSVD/train_C3D_fc8_ImageNet_Places', 'MSVD/val_C3D_fc8_ImageNet_Places'};

%mat_list = {{'Flickr8k/Flicker8k_Dataset/test_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/test_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/train_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/train_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/val_GoogleNet_ImageNet', 'Flickr8k/Flicker8k_Dataset/val_GoogleNet_Places'}};
%out_list = {'Flickr8k/Flicker8k_Dataset/test', 'Flickr8k/Flicker8k_Dataset/train', 'Flickr8k/Flicker8k_Dataset/val'};

%mat_list = {{'Flickr8k/Flicker8k_Dataset/test_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/test_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/train_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/train_GoogleNet_Places'}, ...
%    {'Flickr8k/Flicker8k_Dataset/val_ImageNet_FisherVectors', 'Flickr8k/Flicker8k_Dataset/val_GoogleNet_Places'}};
%out_list = {'Flickr8k/Flicker8k_Dataset/test_KCNN_combined', 'Flickr8k/Flicker8k_Dataset/train_KCNN_combined', 'Flickr8k/Flicker8k_Dataset/val_KCNN_combined'};

%mat_list = {{'VQA/Images/mscoco/train2014/Features_KCNN/test_ImageNet_FisherVectors'}, {'VQA/Images/mscoco/train2014/Features_KCNN/train_ImageNet_FisherVectors'}, {'VQA/Images/mscoco/train2014/Features_KCNN/val_ImageNet_FisherVectors'}};
%out_list = {'VQA/Features/mscoco/test_objectsKCNN', 'VQA/Features/mscoco/train_objectsKCNN', 'VQA/Features/mscoco/val_objectsKCNN'};

mat_list = {{'VQA/Features/mscoco/test2015/GoogleNet_ImageNet'}, {'VQA/Features/mscoco/train2014/GoogleNet_ImageNet'}, {'VQA/Features/mscoco/val2014/GoogleNet_ImageNet'}};
out_list = {'VQA/Features/mscoco/test_objectsCNN', 'VQA/Features/mscoco/train_objectsCNN', 'VQA/Features/mscoco/val_objectsCNN'};


version = 'photo'; % 'photo' or 'video'


data_augmentation = 1;
% data_augmentation = 4;

%suffix = '_Conv_v1';
%suffix = '_Conv_v2';
%suffix = '_dataAugm4';
suffix = '';




nList = length(mat_list);
parfor l = 1:nList
    mat = mat_list{l};
    out = out_list{l};
    nMat = length(mat);

    convert_list = [mat{1} suffix];
    for lm = 2:nMat convert_list = [convert_list ' and ' mat{lm} suffix]; end
    disp(['Converting ' convert_list]);
    
    % Open .mat
    f = {};
    for lm = 1:nMat
	f{lm} = load([path_mat '/' mat{lm} suffix '.mat']);
    end
    
    % If we have cells, then we will store a blank line between each matrix
    % in the cell
    if(iscell(f{1}.features))
        grouping = true;
	
	if(strcmp(version, 'video'))
		% NEW VERSION
		f_counts = fopen([path_mat '/' out suffix '_counts.txt'], 'w');
	end
        

	nGroups = length(f{1}.features);

	% Get number of features per each data augmentation batch
	nFeatures = {};
	for lm = 1:nMat
            nFeatures{lm} = size(f{lm}.features{1},2) / data_augmentation;
	end

    else
        grouping = false;
        nGroups = 1;

        % Get number of features per each data augmentation batch
	nFeatures = {};
        for lm = 1:nMat
	    nFeatures{lm} = size(f{lm}.features,2) / data_augmentation;
	end

    end
    

    % Write each set of data at the bottom of each other (only if data_augmentation > 1)
    for da = 1:data_augmentation
	da_range = {};
	for lm = 1:nMat
 	    da_range{lm} = 1+(da-1)*nFeatures{lm}:da*nFeatures{lm};
	end

        % Write each group in the .csv with a blank space between them
    	for g = 1:nGroups
            if(grouping)
		minSamples = 9999999999;
		for lm = 1:nMat
		    minSamples = min(minSamples, size(f{lm}.features{g}, 1));
		end

		features = [];
		for lm = 1:nMat
            	    features = [features f{lm}.features{g}(1:minSamples,da_range{lm})];
		end
	    	
		if(strcmp(version, 'video'))
	    	    % NEW VERSION
	    	    fprintf(f_counts, [num2str(minSamples) '\n']); % stores number of consecutive vectors corresponding to each video
	    	end

            else
		minSamples = 9999999999;
                for lm = 1:nMat
                    minSamples = min(minSamples, size(f{lm}.features, 1));
                end

		features = [];
                for lm = 1:nMat
            	    features = [features f{lm}.features(1:minSamples,da_range{lm})];
		end
            end
        
            % Write data in csv
            if(g == 1 && da == 1)
            	dlmwrite([path_mat '/' out suffix '_features.csv'], features);
            else
		% OLD VERSION
%               dlmwrite([path_mat '/' out suffix '_features.csv'], 'E', '-append');

	    	if(strcmp(version, 'video'))
            	    dlmwrite([path_mat '/' out suffix '_features.csv'], features, '-append');
	    	end
            end
    	end

    end

    if(strcmp(version, 'video'))
    	% NEW VERSION
    	fclose(f_counts);
    end
end

disp('Done');
exit;
