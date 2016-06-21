
%%%%%%% Extraction of CNN features from object proposals

%% Parameters

%%% Data parameters
% images_path = '.';
% folders = {'test_images'};
% formats = {'.jpg'};

%%%%%%% Flickr8k
% images_path = '/media/HDD_2TB/DATASETS';
% folders = {'Flickr8k/Flicker8k_Dataset'};
% formats = {'.jpg'};

%%%%%%% Flickr8k
images_path = '/media/HDD_2TB/DATASETS';
folders = {'Flickr30k/flickr30k-images'};
formats = {'.jpg'};

%%%%%%% MSVD
% images_path = '/media/HDD_2TB/DATASETS';
% folders = {};
% formats = {};
% 
% msrvd = 'MSVD/Images';
% folders_msrvd = dir([images_path '/' msrvd '/*']);
% folders_msrvd = folders_msrvd(arrayfun(@(x) x.name(1) ~= '.' && isdir([images_path '/' msrvd '/' x.name]), folders_msrvd));
% nNames = length(folders_msrvd);
% for n = 1:nNames
%        names{n} = [msrvd '/' folders_msrvd(n).name];
% end
% folders = {folders{:}, names{:}};
% nFolders = length(folders);
% for f = 1:nFolders
%        formats{f} = '.jpg';
% end


% Images subsampling for reducing the need of computational resources
subsampling = 1; % if subsampling=10 we will only process 1 out of 10 images

% Folder where to start extracting features
offset = 1;



object_proposals_name = 'EdgeBoxes'; % object proposals method used
nProposalsUse = 100; % number of object proposals used per image
rot = [0, 45, 90, 135, 180, 225, 270, 315];

%%% Caffe parameters
CNN_params.caffe_path = '/usr/local/caffe-master2/matlab/caffe';
CNN_params.use_gpu = 1;
IMAGE_DIM = 256;

%%% GoogleNet - ImageNet parameters
CNN_params_ImageNet.batch_size = 30; % Depending on the deploy net structure!!
CNN_params_ImageNet.model_def_file = '/media/HDD_2TB/CNN_MODELS/GoogleNet/deploy_feature_extraction.prototxt';
CNN_params_ImageNet.model_file = '/media/HDD_2TB/CNN_MODELS/GoogleNet/bvlc_googlenet.caffemodel';
CNN_params_ImageNet.input_size = 224;
CNN_params_ImageNet.image_mean = zeros(256, 256, 3);
CNN_params_ImageNet.image_mean(:,:,1) = 104; CNN_params_ImageNet.image_mean(:,:,2) = 117; CNN_params_ImageNet.image_mean(:,:,3) = 123;
CNN_params_ImageNet.size_features = 1024;
CNN_params_ImageNet.features_name = 'GoogleNet_ImageNet_ACL_16_task1';

%%% GoogleNet - Places parameters
CNN_params_Places.batch_size = 10; % Depending on the deploy net structure!!
CNN_params_Places.model_def_file = '/media/HDD_2TB/CNN_MODELS/Places_CNN/googlenet/deploy_feature_extraction.prototxt';
CNN_params_Places.model_file = '/media/HDD_2TB/CNN_MODELS/Places_CNN/googlenet/googlelet_places205_train_iter_2400000.caffemodel';
CNN_params_Places.input_size = 224;
load(['/media/HDD_2TB/CNN_MODELS/Places_CNN/placesCNN/places205CNN_mean.mat']); % image_mean
CNN_params_Places.image_mean = permute(image_mean, [2 3 1]);
%CNN_params.size_features = 205;
CNN_params_Places.size_features = 1024;
CNN_params_Places.features_name = 'GoogleNet_Places';


this_path = pwd;
addpath(CNN_params.caffe_path);


%% Process each folder
nFolders = length(folders);
for f = offset:nFolders
    disp(['Start extraction for folder ' folders{f}]);
    
    % List images
    list_images = dir([images_path '/' folders{f} '/*' formats{f}]);
    list_images = list_images(arrayfun(@(x) x.name(1) ~= '.', list_images));
    list_images = {list_images(:).name};
    nImages = length(list_images);
    selected_images = round(linspace(1,nImages, nImages/subsampling));
    list_images = {list_images{selected_images}};
    nImages = length(list_images);
    
    % Load object proposals
    op = load([images_path '/' folders{f} '/object_proposals_' object_proposals_name '.mat']);
    op = op.object_proposals;

    % Create folders for storing features
    path_ImageNet = [images_path '/' folders{f} '/' CNN_params_ImageNet.features_name];
    path_Places = [images_path '/' folders{f} '/' CNN_params_Places.features_name];
    if(~exist(path_ImageNet))
	mkdir(path_ImageNet);
    end 
    if(~exist(path_Places))
        mkdir(path_Places);
    end

    %% Process each image separately
    for i = 1:nImages
        
	% Load image (if error initialize to matrix of 0s)
	try
		img_mat = imread([images_path '/' folders{f} '/' list_images{i}]);
	catch
		img_mat = zeros(500, 500, 3);
	end
        
	% Get top nProposalsUse object proposals
        this_op = repmat(op(1:nProposalsUse, 1:4, i), length(rot), 1);
	this_op = [this_op reshape(repmat(1:length(rot), nProposalsUse, 1), length(rot)*nProposalsUse, 1)];
	is_empty = repmat(any(this_op==0,2), length(rot), 1);
	
	% Initialize features matrix
	features_ImageNet = zeros(nProposalsUse, CNN_params_ImageNet.size_features);
	features_Places = zeros(nProposalsUse, CNN_params_Places.size_features);

	cd(CNN_params.caffe_path)

	%% Process ImageNet
	batch_size = CNN_params_ImageNet.batch_size;
    	matcaffe_init(CNN_params.use_gpu, CNN_params_ImageNet.model_def_file, CNN_params_ImageNet.model_file); % initialize using or not GPU and model/network files

	tot_samples = nProposalsUse*length(rot);
	for ib = 1:batch_size:tot_samples
            this_batch = ib:min(ib+batch_size-1, tot_samples);
            im_list = cell(1,batch_size);
            [im_list{:}] = deal(0);
            count = 1;
            for j = this_batch
		if(is_empty(j))
			im_list{count} = zeros(1,1);
		else
			im_list{count} = img_mat(this_op(j,2):this_op(j,2)+this_op(j,4)-1, this_op(j,1):this_op(j,1)+this_op(j,3)-1, :);
		end
		this_rot = this_op(j,5);
                count = count+1;
            end

	    CROPPED_DIM = CNN_params_ImageNet.input_size;
	    num_images = length(im_list);
	    images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

	    for ip=1:num_images
		im = im_list{ip};

            	% resize to fixed input size
            	im = single(im);
            	im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');

		% Transform GRAY to RGB
                if size(im,3) == 1
                    im = cat(3,im,im,im);
                end

		% permute from RGB to BGR (IMAGE_MEAN is already BGR)
                im = im(:,:,[3 2 1]) - CNN_params_ImageNet.image_mean;

		% ROTATE AND CROP
		im = imrotate(im, rot(this_rot));

		% if the rotation is even we will have to cut a smaller portion and then resize
		if(mod(this_rot, 2) == 0)
                    CROPPED_DIM = floor(sqrt((IMAGE_DIM/2)^2 * 2));
            	else
                    CROPPED_DIM = CNN_params_ImageNet.input_size;
            	end
		indices = [0 size(im,1)-CROPPED_DIM] + 1;
            	center = floor(indices(2) / 2)+1;

            	% Crop the center of the image
            	im = permute(im(center:center+CROPPED_DIM-1, center:center+CROPPED_DIM-1,:),[2 1 3]);
		images(:,:,:,ip) = imresize(im, [CNN_params_ImageNet.input_size CNN_params_ImageNet.input_size], 'bilinear');
	    end
            scores = caffe('forward', {images});
            scores = squeeze(scores{1});
            features_ImageNet(this_batch, :) = scores(:,1:length(this_batch))';
        end

% 	%% Process Places
% 	batch_size = CNN_params_Places.batch_size;
%         matcaffe_init(CNN_params.use_gpu, CNN_params_Places.model_def_file, CNN_params_Places.model_file); % initialize using or not GPU and model/network files
%         for ib = 1:batch_size:tot_samples
%             this_batch = ib:min(ib+batch_size-1, tot_samples);
%             im_list = cell(1,batch_size);
%             [im_list{:}] = deal(0);
%             count = 1;
%             for j = this_batch
%                 if(is_empty(j))
%                         im_list{count} = zeros(1,1);
%                 else
%                 	im_list{count} = img_mat(this_op(j,2):this_op(j,2)+this_op(j,4)-1, this_op(j,1):this_op(j,1)+this_op(j,3)-1, :);
% 		end
% 		this_rot = this_op(j,5);
%                 count = count+1;
%             end
%             
% 	    CROPPED_DIM = CNN_params_Places.input_size;
%             num_images = length(im_list);
%             images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');
% 
%             parfor ip=1:num_images
%                 im = im_list{ip};
% 
%                 % resize to fixed input size
%                 im = single(im);
%                 im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% 
% 		% Transform GRAY to RGB
%                 if size(im,3) == 1
%                     im = cat(3,im,im,im);
%                 end
% 
% 		% permute from RGB to BGR (IMAGE_MEAN is already BGR)
%                 im = im(:,:,[3 2 1]) - CNN_params_Places.image_mean;
% 
%                 % ROTATE AND CROP
%                 im = imrotate(im, rot(this_rot));
% 
%                 % if the rotation is even we will have to cut a smaller portion and then resize
%                 if(mod(this_rot, 2) == 0)
%                     CROPPED_DIM = floor(sqrt((IMAGE_DIM/2)^2 * 2));
%                 else
%                     CROPPED_DIM = CNN_params_Places.input_size;
%                 end
%                 indices = [0 size(im,1)-CROPPED_DIM] + 1;
%                 center = floor(indices(2) / 2)+1;
% 
%                 % Crop the center of the image
%                 im = permute(im(center:center+CROPPED_DIM-1, center:center+CROPPED_DIM-1,:),[2 1 3]);
%                 images(:,:,:,ip) = imresize(im, [CNN_params_Places.input_size CNN_params_Places.input_size], 'bilinear');
%             end
% 	    scores = caffe('forward', {images});
%             scores = squeeze(scores{1});
%             features_Places(this_batch, :) = scores(:,1:length(this_batch))';
%         end

	cd(this_path);
        
	%% Save results current image
	save([path_ImageNet '/' list_images{i} '_ImageNet.mat'], 'features_ImageNet');
%         save([path_Places '/' list_images{i} '_Places.mat'], 'features_Places');

	%% Show progress
	if(mod(i, 10)==0 || i == nImages)
		disp(['Processed ' num2str(i) '/' num2str(nImages) ' images.']);
	end
    end

end

disp('Done');
exit;
