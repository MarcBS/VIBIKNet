
%folder_path = '/media/HDD_2TB/DATASETS/MSRVD';
folder_path = '/media/HDD_2TB/DATASETS/MSVD';

images = 'Images';

% train, val, test
split = {'train_list.txt', 'val_list.txt', 'test_list.txt'};

order = {'train', 'val', 'test'};


%features_list = {'GoogleNet_ImageNet.mat', 'GoogleNet_Places.mat'};
%features_list = {'C3D_features.mat'};
features_list = {'C3D_features_fc8.mat'};
coordinates_list = {};

% features_list = {'GoogleNet_ImageNet_Conv_v1.mat', 'GoogleNet_Places_Conv_v1.mat', 'GoogleNet_ImageNet_Conv_v2.mat', 'GoogleNet_Places_Conv_v2.mat'};
% coordinates_list = {'GoogleNet_ImageNet_Conv_v1_coordinates.mat', 'GoogleNet_Places_Conv_v1_coordinates.mat', 'GoogleNet_ImageNet_Conv_v2_coordinates.mat', 'GoogleNet_Places_Conv_v2_coordinates.mat'};


nFeat = length(features_list);
nSplits = length(split);

%% Process each split
for s = 1:nSplits
    disp(['Starting split ' order{s}]);
    
    images_split = fileread([folder_path '/' split{s}]);
    images_split = regexp(images_split, '\n', 'split');
    nLines = length(images_split)-1;

    %% Load each features matrix
    for f = 1:nFeat
        disp(['Storing features ' features_list{f}]);
       
	features = {};
	coordinates = {};
	for v = 1:nLines
		features_all = load([folder_path '/' images '/' images_split{v} '/' features_list{f}]);
        	features{v} = features_all.features;

		if(length(coordinates_list) >= f && ~isempty(coordinates_list{f}))
			coordinates_all = load([folder_path '/' images '/' images_split{v} '/' coordinates_list{f}]);
			coordinates{v} = coordinates_all.coordinates;
			has_coordinates = true;
		else
			has_coordinates = false;
		end
	end

	if(has_coordinates)
		save([folder_path '/' order{s} '_' coordinates_list{f}], 'coordinates');
	end
        save([folder_path '/' order{s} '_' features_list{f}], 'features', '-v7.3');
    end
    
end


disp('Done');
exit;
