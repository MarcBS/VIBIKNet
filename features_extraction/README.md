
- Requirements
	- Matlab
	- Caffe compiled
	- Caffe's Matlab wrapper compiled '/path/to/caffe/matlab'

- Extract Features from last Fully-Connected layer (usual feature extraction)

	1) In mainExtractFCLayer change the following parameters:
	
		- CNN_params.caffe_path: path to caffe's matlab wrapper
		- data_path: path where all the image folders are stored
		- folders: cell array with list of folders to process
		- formats: iamge formats for each of the folders
		- CNN_params.model_def_file: path to each network model file (one for Places and the other for ImageNet)
		- CNN_params.model_file: path to the trained model weights

	2) Run mainExtractFCLayer.m --> will extract the CNN features for all the folders

	3) In indexFeatures.m change the corresponding parameters for pointing to the currently extracted features, the 
		{train, val, test} image lists, etc. Don't use (empty cell) the variable coordinates_list.

	4) Run indexFeatures.m --> will reformat the extracted features for matching the indexing of the corresponding 
		{train, val, test} image lists

- Extract Features from last Convolutional layer (attention mechanism)
	
	1) Apply the same steps 1) and 2) when using the fully-connected layer as a feature extractor, but on the 
		file mainExtractConvLayer.m

	2) Apply the same steops 3) and 4) when using the fully-connected layer as a feature extractor, but this time
		we must make sure that we are indexing both ...Conv_v1.mat and ...Conv_v2.mat features in features_list
		and their corresponding coordinates in coordinates_list.

- Extract KCNN Features

	1) See README in KCNN folder

- Convert .mat features into .csv

	1) Change the corresponding parameters in mat2csv.m. Pointing to the folders paths with path_mat, the .mat features
		with mat_list and the output paths with out_list. Change also the suffix if needed!

	2) Run mat2csv.m


- Apply PCA on features stored in .csv

	1) Change the corresponding parameters in applyNormPCA.m pointing to the .csv folders with the traning features.
		Run the script for extracting the necessari PCA parameters from the training set.
        2) Change the corresponding parameters in applyNormPCA_test.m pointing to the .csv folders with train/val/test
		features. Run the script for extracting a set of features with their dimensionality reduced.
