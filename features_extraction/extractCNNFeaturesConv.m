function [ features_folders, coordinates ] = extractCNNFeatures( data_path, folders, formats, CNN_params )
%EXTRACTCNNFEATURESS Extract global CNN features for the given folders

    %% Prepare paths and initialize caffe
    batch_size = CNN_params.batch_size;

    this_path = pwd;

    %addpath(CNN_params.caffe_path);
    %cd(CNN_params.caffe_path)
    %matcaffe_init(CNN_params.use_gpu, CNN_params.model_def_file, CNN_params.model_file); % initialize using or not GPU and model/network files


    %% Initialize caffe
    %%% NEW VERSION
    addpath([CNN_params.caffe_path '/matlab']);
    if(CNN_params.use_gpu)
        caffe.set_mode_gpu();
        caffe.set_device(CNN_params.gpu_id);
    else
        caffe.set_mode_cpu();
    end
    caffe.reset_all(); % reset the used net from caffe
    net = caffe.Net(CNN_params.model_def_file, CNN_params.model_file, 'test');


    nFold = length(folders);
    count_fold = 1;
    tic;
    features_folders = {};
    coordinates = {};
    for f = folders
        format = formats{count_fold}; 
        images = dir([data_path '/' f{1} '/*' format]);
        features = zeros(length(images), CNN_params.size_features);
	features_v2 = zeros(length(images), CNN_params.size_features);
	coord = cell(1, length(images));
	coord_v2 = cell(1, length(images));
        %% For each image in this folder
        count_im = 1;
        names = {images(:).name};
        nImages = length(names);

        if(nImages == 0)
            error(['Images with format ' format ' not found in folder ' f{1} '.']);
        end

        for i = 1:batch_size:nImages
            this_batch = i:min(i+batch_size-1,  nImages);
            im_list = cell(1,batch_size);
            [im_list{:}] = deal(0);
            count = 1;
            for j = this_batch
                im_list{count} = [data_path '/' f{1} '/' names{j}];
                count = count+1;
            end
            images = {prepare_batch_custom_crop(im_list, CNN_params.input_size, CNN_params.image_mean, batch_size, 1, false)};      
            %scores = caffe('forward', images);
            scores = net.forward(images);
            scores = squeeze(scores{1});
            
	    for j = 1:length(this_batch)
		s_j = scores(:,:,:,j);

		% Sum scores along third dimension
		sums = sum(s_j, 3);
		% Get indices from reshaped matrix
	 	[y_ind, x_ind] = ind2sub([size(s_j,1) size(s_j,2)], 1:size(s_j,1)*size(s_j,2));
		% Reshape and sort matrix
		s_sort = reshape(sums, [1, size(s_j,1)*size(s_j,2)]);
		[s_sort, s_pos] = sort(s_sort, 'descend');

		% Get top 1
		max_y = y_ind(s_pos(1));
		max_x = x_ind(s_pos(1));
		
		% Get max(top 5)
		max_y_top = y_ind(s_pos(1:5));
		max_x_top = x_ind(s_pos(1:5));

		features(this_batch(j), :) = reshape(s_j(max_y, max_x, :), [1 size(s_j,3)]);
		vecs = zeros(5, CNN_params.size_features);
		for k = 1:5
			vecs(k,:) = reshape(s_j(max_y_top(k), max_x_top(k), :), [1 size(s_j,3)]);
		end
		features_v2(this_batch(j), :) = max(vecs); 
		coord{this_batch(j)} = {[max_y max_x]};
		coord_v2{this_batch(j)} = {[max_y_top' max_x_top']};
	    end
          
            %% Show progress
            if(mod(i, 51) == 0 || i == nImages)
                disp(['    Processed ' num2str(i) '/' num2str(nImages) ' images...']);
            end

        end

        features_folders{count_fold} = {features, features_v2};
	coordinates{count_fold} = {coord, coord_v2};
        clear features;
        count_fold = count_fold+1;
    end
    toc
    cd(this_path)


end

