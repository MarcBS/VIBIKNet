
%%%%%%% Extraction of Kernel-KCNN features from a set of images


%% Parameters
debug = true;


%%% Add some required paths
piotr_dollar_toolbox_path = '../../piotr_toolbox';
edge_boxes_path = '../../edges';
yael_path = '../../yael';
inria_fisher_path = '../../inria_fisher_v1';

addpath(genpath(piotr_dollar_toolbox_path));
addpath(edge_boxes_path);
addpath(yael_path);
addpath(inria_fisher_path);

%%% Data parameters
images_path = '.';
folders = {'test_images'};
formats = {'.jpg'};

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

%%% Fisher Vector parameters



%% Process each folder
nFolders = length(folders);
for f = 1:nFolders
    
    % List images
    list_images = dir([images_path '/' folders{f} '/*' formats{f}]);
    list_images = list_images(arrayfun(@(x) x.name(1) ~= '.', list_images));
    list_images = {list_images(:).name};
    nImages = length(list_images);
    
    features = {};
    %% Process each image separately
    for i = 1:nImages
        
        % Apply Edge Boxes on each image
        img = imread([images_path '/' folders{f} '/' list_images{i}]);
        bbs=edgeBoxes(img, model_edgeboxes, opts_edgeboxes);
        
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
        
    end
    
end


