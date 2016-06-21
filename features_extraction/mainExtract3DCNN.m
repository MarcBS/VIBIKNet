
%% Example Data
data_path = '/media/HDD_2TB/DATASETS';
%folders = {'COCO/test2014', 'COCO/val2014', 'COCO/train2014', 'Flickr8k/Flicker8k_Dataset', 'Flickr30k/flickr30k-images'};
folders = {};
formats = {};

%folders = {'Flickr8k/Flicker8k_Dataset'};
%formats = {'.jpg'};


%% Add folders from MSRVD too
%msrvd = 'MSRVD/Images';

msrvd = 'MSVD/Images';
folders_msrvd = dir([data_path '/' msrvd '/*']);
folders_msrvd = folders_msrvd(arrayfun(@(x) x.name(1) ~= '.' && isdir([data_path '/' msrvd '/' x.name]), folders_msrvd));
nNames = length(folders_msrvd);
for n = 1:nNames
	names{n} = [msrvd '/' folders_msrvd(n).name];
end
folders = {folders{:}, names{:}};
nFolders = length(folders);
for f = 1:nFolders
	formats{f} = '.jpg';
end


%% CNN Parameters
CNN_params.caffe_path = '/usr/local/C3D';
CNN_params.use_gpu = 1;

%%% C3D network  parameters
addpath('/media/HDD_2TB/CNN_MODELS/Conv3D-Sport1M/');
CNN_params.batch_size = 10;
CNN_params.model_def_file = '/media/HDD_2TB/CNN_MODELS/Conv3D-Sport1M/c3d_sport1m_feature_extractor_frm.prototxt';
CNN_params.model_file = '/media/HDD_2TB/CNN_MODELS/Conv3D-Sport1M/conv3d_deepnetA_sport1m_iter_1900000.caffemodel';
CNN_params.mean_file = '/media/HDD_2TB/CNN_MODELS/Conv3D-Sport1M/sport1m_train16_128_mean.binaryproto';

% CNN_params.features_size = 4096;
% CNN_params.layer_name = 'fc7-1';

CNN_params.features_size = 487;
CNN_params.layer_name = 'fc8-1';

% CNN_params.num_data_augmentation = 1;
features_name = 'C3D_features_fc8';


%% Create temporal folder
tmp_folder = 'tmp';
if(exist(tmp_folder))
    rmdir(tmp_folder, 's');
end
mkdir(tmp_folder);


%% Extract features
disp(' ');
disp('Extracting 3D-CNN features.');
disp(' ');
nFolders = length(folders);
for f = 1:nFolders
	disp(['Extracting folder ' folders{f} ' (' num2str(f) '/' num2str(nFolders) ')']);

	%% Prepare images list
	out_features = [data_path '/' folders{f} '/' features_name];
	if(~exist(out_features))
	    mkdir(out_features);	
	end

	img_list = dir([data_path '/' folders{f} '/*' formats{f}]);
	img_list = img_list(arrayfun(@(x) x.name(1) ~= '.', img_list));
	list_file = [pwd '/' tmp_folder '/imgs_list.txt'];
	out_list_file = [pwd '/' tmp_folder '/imgs_list_out.txt'];
	nImgs = length(img_list);
	fid = fopen(list_file, 'w');
	fid_out = fopen(out_list_file, 'w');

	% REMOVE THE LAST 16-1 IMAGES (WE NEED A WINDOW OF 16 TO CALCULATE THE FEATURES)
	nImgs = max(0, nImgs-15);	

	for i = 1:nImgs
	    [~, im_name, ~] = fileparts(img_list(i).name);
	    fprintf(fid, [data_path '/' folders{f} '/ ' num2str(i) ' 0\n']);
	    fprintf(fid_out, [out_features '/' im_name '\n']);
	end
	fclose(fid);
	fclose(fid_out);


	%% Prepare input structure
	struct = fileread(CNN_params.model_def_file);
	CNN_params.model_def_file = [pwd '/' tmp_folder '/frames_struct_tmp.prototxt'];
	struct = regexprep(struct, 'source: "\w+[\W+\w+]+?"', ['source: "' list_file '"']);
	struct = regexprep(struct, 'mean_file: "\w+[\W+\w+]+?"', ['mean_file: "' CNN_params.mean_file '"']);	
	struct = regexprep(struct, 'batch_size: \d+', ['batch_size: ' num2str(CNN_params.batch_size)]);
	fid_struct = fopen(CNN_params.model_def_file, 'w');
	fprintf(fid_struct, struct);
	fclose(fid_struct);


	%% Prepare features extraction script
	text = fileread('mainExtract3DCNN.sh');
	path_text = [tmp_folder '/mainExtract3DCNN_tmp.sh'];
	
	text = regexprep(text,'%path_structure%', CNN_params.model_def_file);
	text = regexprep(text,'%path_weights%', CNN_params.model_file);
	if(CNN_params.use_gpu)
	    text = regexprep(text,'%gpu%', '0');
	else
	    text = regexprep(text,'%gpu%', '-1');
	end
	text = regexprep(text,'%batch_size%', num2str(CNN_params.batch_size));
	text = regexprep(text,'%num_batches%', num2str(ceil(nImgs/CNN_params.batch_size)));
	text = regexprep(text,'%list_images%', out_list_file);
	text = regexprep(text,'%layer_name%', CNN_params.layer_name);
	text = regexprep(text,'%caffe_path%', CNN_params.caffe_path);

	fid = fopen(path_text, 'w');
	fprintf(fid, text);
	fclose(fid);
	system(['chmod u+rwx ' path_text]);


        %% Run CNN and extract features
	tmp_output = [tmp_folder '/out_extract3D.txt'];
	disp(['    Check CNN output in ' tmp_output]);
	[out, msg] = system(['./' path_text ' >' tmp_output ' 2>&1']);
	if(out > 0)
	    disp('Execution terminated with error:');
	    error(msg);
	end

	
	%% Store features
	features = zeros(nImgs, CNN_params.features_size);
	for i = 1:nImgs
	    [~, im_name, ~] = fileparts(img_list(i).name);
	    [~, features(i, :)]=read_binary_blob([out_features '/' im_name '.' CNN_params.layer_name]);
	end
 	save([data_path '/' folders{f} '/' features_name '.mat'], 'features');
 	disp(' ');
	rmdir(out_features, 's');

end

disp('Done');
exit;
