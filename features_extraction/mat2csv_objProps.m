
%% This script converts .mat files into .csv files
path_mat = '/media/HDD_2TB/DATASETS';

mat_list = {'VQA/Images/mscoco/test2015/GoogleNet_ImageNet', 'VQA/Images/mscoco/train2014/GoogleNet_ImageNet', 'VQA/Images/mscoco/val2014/GoogleNet_ImageNet'};
out_list = {'VQA/Features/mscoco/test_objProps', 'VQA/Features/mscoco/train_objProps', 'VQA/Features/mscoco/val_objProps'};
lists = {'VQA/Images/mscoco/test_list.txt', 'VQA/Images/mscoco/train_list.txt', 'VQA/Images/mscoco/val_list.txt'};


suffix = '_ImageNet.mat';
rotations_per_proposal = 8; % number of rotations obtained per proposal. Only the first one will be used.


version = 'photo'; % 'photo' or 'video'
break_img = 1000


nList = length(mat_list);
for l = 1:nList
    mat = mat_list{l};
    out = out_list{l};
    list = lists{l};

    convert_list = [mat '*' suffix];
    disp(['Converting ' convert_list]);
   
    list = fileread([path_mat '/' list]);
    list = regexp(list, '\n', 'split');
    if(isempty(list{end}))
        list = {list{1:end-1}};
    end
    nIm = length(list);

    part = 1
    prev_part = 0;
    for i = 1:nIm
        im = list{i};
        f = load([path_mat '/' mat '/' im suffix]);
        f = f.features_ImageNet;

        nProps = size(f,1)/rotations_per_proposal;
        nFeat = size(f,2);
        feat = zeros(1, nFeat*nProps);
        for p = 1:nProps
            feat((p-1)*nFeat+1:p*nFeat) = f((p-1)*rotations_per_proposal+1,:);
        end

        if(prev_part ~= part)
            dlmwrite([path_mat '/' out '_features' num2str(part) '.csv'], feat);
        else
            dlmwrite([path_mat '/' out '_features' num2str(part) '.csv'], feat, '-append');
        end

        prev_part = part;
        if(mod(i,break_img) == 0)
            part = part+1;
        end
    end 

end

disp('Done');
exit;
