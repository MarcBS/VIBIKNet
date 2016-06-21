% ------------------------------------------------------------------------
function images = prepare_batch_custom_crop(image_files,CROPPED_DIM,IMAGE_MEAN,batch_size,data_augm, imgs_loaded)
% ------------------------------------------------------------------------
if nargin < 2
    CROPPED_DIM = 227;
end
if nargin < 3
    d = load('ilsvrc_2012_mean');
    IMAGE_MEAN = d.image_mean; 
end
num_images = length(image_files);
if nargin < 4
    batch_size = num_images;
end
if nargin < 5
    data_augm = 1;
end

IMAGE_DIM = 256;
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;

if(data_augm == 1)
    center = floor(indices(2) / 2)+1;
    up = center;
    left = center;
% choose a random-positioned crop for "data_augmentation"
else
    max_crop = IMAGE_DIM-CROPPED_DIM+1;
    up = randsample(1:max_crop, 1);
    left = randsample(1:max_crop, 1);
end

num_images = length(image_files);
images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

parfor i=1:num_images
    % read file
%     fprintf('%c Preparing %s\n',13,image_files{i});
    try
        %im = imread(image_files{i});
        
	if(~imgs_loaded)
                im = imread(image_files{i});
        else
                im = image_files{i};
        end

	% resize to fixed input size
        im = single(im);
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        % permute from RGB to BGR (IMAGE_MEAN is already BGR)
        im = im(:,:,[3 2 1]) - IMAGE_MEAN;
        % Crop the center of the image
        images(:,:,:,i) = permute(im(up:up+CROPPED_DIM-1,...
            left:left+CROPPED_DIM-1,:),[2 1 3]);
    catch
        warning('Problems with file',image_files{i});
    end
end
