
%%%%%%% Extraction of Object proposals from images


%% Parameters
debug = false;

%%% Add some required paths
piotr_dollar_toolbox_path = '../../piotr_toolbox';
edge_boxes_path = '../../edges';

addpath(genpath(piotr_dollar_toolbox_path));
addpath(edge_boxes_path);

%%% Data parameters
% images_path = '.';
% folders = {'test_images'};
% formats = {'.jpg'};


%%%%%%% Flickr8k
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr8k/Flicker8k_Dataset'};
%formats = {'.jpg'};


%%%%%% Flickr30k
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {'Flickr30k/flickr30k-images'};
%formats = {'.jpg'};
%store_paths = {[images_path '/Flickr30k/ACL_16_task1']};

%%%%%%% MSVD
%images_path = '/media/HDD_2TB/DATASETS';
%folders = {};
%formats = {};
%
%msrvd = 'MSVD/Images';
%folders_msrvd = dir([images_path '/' msrvd '/*']);
%folders_msrvd = folders_msrvd(arrayfun(@(x) x.name(1) ~= '.' && isdir([images_path '/' msrvd '/' x.name]), folders_msrvd));
%nNames = length(folders_msrvd);
%for n = 1:nNames
%       names{n} = [msrvd '/' folders_msrvd(n).name];
%end
%folders = {folders{:}, names{:}};
%nFolders = length(folders);
%for f = 1:nFolders
%       formats{f} = '.jpg';
%end

%%%%%%% VQA (mscoco)
images_path = '/media/HDD_2TB/DATASETS/VQA';
%folders = {'Images/mscoco/train2014', 'Images/mscoco/val2014', 'Images/mscoco/test2015'};
%folders = {'Images/mscoco/val2014'};
folders = {'Images/mscoco/test2015'};
formats = {'.jpg', '.jpg', '.jpg'};
%store_paths = {[images_path '/Features/mscoco/train2014'], [images_path '/Features/mscoco/val2014'], [images_path '/Features/mscoco/test2015']};
%store_paths = {[images_path '/Features/mscoco/val2014']};
store_paths = {[images_path '/Features/mscoco/test2015']};


%%% EdgeBoxes object proposal parameters
% load pre-trained edge detection model and set opts (see edgesDemo.m)
model_edgeboxes=load([edge_boxes_path '/models/forest/modelBsds']); model_edgeboxes=model_edgeboxes.model;
model_edgeboxes.opts.multiscale=0; model_edgeboxes.opts.sharpen=2; model_edgeboxes.opts.nThreads=4;

% set up opts for edgeBoxes (see edgeBoxes.m)
opts_edgeboxes = edgeBoxes;
opts_edgeboxes.alpha = .65;     % step size of sliding window search
opts_edgeboxes.beta  = .75;     % nms threshold for object proposals
opts_edgeboxes.minScore = .01;  % min score of boxes to detect
opts_edgeboxes.maxBoxes = 500;  % max number of boxes to detect



%% Process each folder
nFolders = length(folders);
for f = 1:nFolders
    disp(' ');
    disp(['Start extraction for folder ' folders{f}]);
    disp(' ');
    
    % List images
    list_images = dir([images_path '/' folders{f} '/*' formats{f}]);
    list_images = list_images(arrayfun(@(x) x.name(1) ~= '.', list_images));
    list_images = {list_images(:).name};
    nImages = length(list_images);
    
    object_proposals = zeros(opts_edgeboxes.maxBoxes, 5, nImages);
    %% Process each image separately
    for i = 1:nImages
        
        % Apply Edge Boxes on each image
        try
            img = imread([images_path '/' folders{f} '/' list_images{i}]);
        catch
            disp('Problem with image:');
            disp([images_path '/' folders{f} '/' list_images{i}]);
            % If there is a problem with an image, then we will process a matrix with 0s
            img = zeros(500, 500, 3);
        end
        if(length(size(img)) == 2)
            % Let's convert it to RGB
            img = repmat(img,[1 1 3]);
        end
        
        bbs=edgeBoxes(img, model_edgeboxes, opts_edgeboxes);
        nExtracted = size(bbs,1); % maybe it extracts less candidates that we where expecting
        
        % Store result in matrix
        object_proposals(1:nExtracted,:,i) = bbs;
        
        % Debug
        if(debug)
            save(['bbs_' list_images{i} '.mat'], 'bbs');
            f = figure;
            image(img);
            for bb = 1:20
                rectangle('Position', bbs(bb,1:4), 'EdgeColor', 'red');
            end
            saveas(f, ['fig_' list_images{i} '.jpg']);
            close(gcf);
        end
        
        %% Show progress
        if(mod(i,50) == 0 || i == nImages)
            disp(['Processed ' num2str(i) '/' num2str(nImages) ' images.']);
        end
    end
    
    %% Save result in folder
    save([store_paths{f} '/object_proposals_EdgeBoxes.mat'], 'object_proposals');
end

disp('Done');
exit;
