
%folder_path = '/media/HDD_2TB/DATASETS/Flickr8k';
folder_path = '/media/HDD_2TB/DATASETS/Flickr30k';

%images = 'Flicker8k_Dataset';
images = 'flickr30k-images';

format = '.jpg';


store_path = [folder_path '/ACL_16_task1'];

path_features = [folder_path '/ACL_16_task1'];

% train, val, test
%split = {'text/Flickr_8k.trainImages.txt', 'text/Flickr_8k.devImages.txt', 'text/Flickr_8k.testImages.txt'};
%split = {'flickr30k-images/train_list.txt', 'flickr30k-images/val_list.txt', 'flickr30k-images/test_list.txt'};
split = {'ACL_16_task1/split/train_images.txt', 'ACL_16_task1/split/val_images.txt', 'ACL_16_task1/split/test_images.txt'};

order = {'train', 'val', 'test'};

%features_list = {'GoogleNet_ImageNet_Conv_v1.mat', 'GoogleNet_Places_Conv_v1.mat', 'GoogleNet_ImageNet_Conv_v2.mat', 'GoogleNet_Places_Conv_v2.mat'};
%coordinates_list = {'GoogleNet_ImageNet_Conv_v1_coordinates.mat', 'GoogleNet_Places_Conv_v1_coordinates.mat', 'GoogleNet_ImageNet_Conv_v2_coordinates.mat', 'GoogleNet_Places_Conv_v2_coordinates.mat'};

%features_list = {'GoogleNet_ImageNet_dataAugm4.mat', 'GoogleNet_Places_dataAugm4.mat'};
features_list = {'GoogleNet_ImageNet.mat', 'GoogleNet_Places.mat'};
coordinates_list = {};



%% Load all images
images_all = dir([folder_path '/' images '/*' format]);
images_all = images_all(arrayfun(@(x) x.name(1) ~= '.', images_all));
images_all = {images_all(:).name};


nFeat = length(features_list);
nSplits = length(split);

%% Process each split
for s = 1:nSplits
    disp(['Starting split ' order{s}]);
    
    images_split = fileread([folder_path '/' split{s}]);
    images_split = regexp(images_split, '\n', 'split');
    nLines = length(images_split)-1;

    sorted = zeros(1, nLines);
    for l = 1:nLines
        pos = find(ismember(images_all, images_split{l}));
        if(isempty(pos) || length(pos) > 1)
            disp(images_split{l})
            disp(pos)
            error('ELEMENT NOT FOUND!');
        end
        sorted(l) = pos;
    end
    
    %% Load each features matrix
    for f = 1:nFeat
        disp(['Storing features ' features_list{f}]);
        
        features_all = load([path_features '/' features_list{f}]);
        features = features_all.features(sorted,:);
 
        save([store_path '/' order{s} '_' features_list{f}], 'features');

	if(length(coordinates_list) >= f && ~isempty(coordinates_list{f}))
        	coordinates_all = load([path_features '/' coordinates_list{f}]);
		coordinates = {coordinates_all.coordinates{sorted}};

		save([store_path '/' order{s} '_' coordinates_list{f}], 'coordinates');

        end

    end
    
end


disp('Done');
exit;
