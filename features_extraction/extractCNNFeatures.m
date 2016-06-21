function features_folders = extractCNNFeatures( data_path, folders, formats, CNN_params )
%EXTRACTCNNFEATURESS Extract global CNN features for the given folders

    %% Prepare paths and initialize caffe
    batch_size = CNN_params.batch_size;

    num_data_augmentation = CNN_params.num_data_augmentation;

    this_path = pwd;
    %addpath(CNN_params.caffe_path);
    %cd(CNN_params.caffe_path)

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

    %matcaffe_init(CNN_params.use_gpu, CNN_params.model_def_file, CNN_params.model_file); % initialize using or not GPU and model/network files

    nFold = length(folders);
    count_fold = 1;
    tic;
    features_folders = {};
    for f = folders
        format = formats{count_fold}; 
        images = dir([data_path '/' f{1} '/*' format]);
	images = images(arrayfun(@(x) x.name(1) ~= '.', images));
        features = zeros(length(images), CNN_params.size_features * num_data_augmentation);
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
	    
	    % Apply once for each 'num_data_augmentation'
	    for data_augm = 1:num_data_augmentation
            	images = {prepare_batch_custom_crop(im_list, CNN_params.input_size, CNN_params.image_mean, batch_size, data_augm, false)};      
            	
		%scores = caffe('forward', images);
                scores = net.forward(images);

            	scores = squeeze(scores{1});

		feat_pos = (data_augm-1)*CNN_params.size_features+1:data_augm*CNN_params.size_features;
            	features(this_batch, feat_pos) = scores(:,1:length(this_batch))';
	    end

	    %% Show progress
	    if(mod(i, 51) == 0 || i == nImages)
		disp(['    Processed ' num2str(i) '/' num2str(nImages) ' images...']);
	    end

        end

        features_folders{count_fold} = features;
        clear features;
        count_fold = count_fold+1;
    end
    toc
    cd(this_path)


end

