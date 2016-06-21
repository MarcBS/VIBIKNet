import json
import numpy as np
import os
from collections import Counter
import operator


#dataset_path = '/home/lvapeab/smt/tasks/image_desc/VQA/'
dataset_path = '/media/HDD_2TB/DATASETS/VQA/'

IMG_FILES = {'train': 'Images/mscoco/train_list.txt', 
                 'val': 'Images/mscoco/val_list.txt', 
                 'test': 'Images/mscoco/test_list.txt'}
IMG_FEATURES = {'train': 'Features/mscoco_objectsKCNN_L2/train2014/objectsKCNN_L2.npy', 
                 'val': 'Features/mscoco_objectsKCNN_L2/val2014/objectsKCNN_L2.npy', 
                 'test': 'Features/mscoco_objectsKCNN_L2/test2015/objectsKCNN_L2.npy'}
QST_FILES = {'train': 'Questions/OpenEnded_mscoco_train2014_questions.json', 
                 'val': 'Questions/OpenEnded_mscoco_val2014_questions.json', 
                 'test': 'Questions/OpenEnded_mscoco_test2015_questions.json'}
ANS_FILES = {'train': 'Annotations/mscoco_train2014_annotations.json',
                 'val': 'Annotations/mscoco_val2014_annotations.json'}             

data_preprocessing = None # None, 'single_frequent', 'most_frequent'
k_frequent = 3 # only used if data_preprocessing=='most_frequent'

force_reload_features = False

# process each data split
for k in IMG_FILES.keys():
    
    print 'Processing '+k+' split.'
    print '==============================='
    
    imgs = IMG_FILES[k]
    img_features = IMG_FEATURES[k]
    out_imgs_ids = imgs.split('.')[0] + '_ids.txt'
    out_imgs_features_store = '/'.join(img_features.split('/')[:-1])
    out_imgs_features_list = imgs.split('.')[0] + '_features.txt'
    
    qsts = QST_FILES[k]
    out_qsts = qsts.split('.')[0] + '.txt'
    out_qsts_ids = qsts.split('.')[0] + '_ids.txt'
    
    if(k in ANS_FILES.keys()):
        anss = ANS_FILES[k]
        out_anss = anss.split('.')[0] + '.txt'
        out_anss_ids = anss.split('.')[0] + '_ids.txt'
    
    # read img IDs 
    print 'Processing images.'
    img_list_names = []
    img_list_ids = []
    img_list_ids_idx = dict()
    with open(dataset_path+imgs, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip('\n')
            line = line.split('.')[0]
            img_list_names.append(line)
            im = int(line.split('_')[-1])
            img_list_ids.append(im)
            img_list_ids_idx[im] = i
       
    # read features
    feats = np.load(dataset_path+img_features).item()
    
    # store img features paths and .npy files
    n_feat = len(feats)
    with open(dataset_path+out_imgs_features_list, 'w') as f:
        for i, name in enumerate(img_list_names):
            im_feat_path = out_imgs_features_store+'/'+name+'.npy'
            if not os.path.isfile(dataset_path+im_feat_path) or force_reload_features:
                np.save(dataset_path+im_feat_path, feats[name])
            f.write(im_feat_path+'\n')
            if i%10000 == 0:
                print '\tStored features for %d/%d images.' % (i, n_feat)
    # store img ids
    with open(dataset_path+out_imgs_ids, 'w') as f:
        for id in img_list_ids:
            f.write(str(id)+'\n')
            
    # read questions and store in img's order
    print 'Processing questions.'
    qst_list = [{'qst_ids': [], 'qst_txt': []} for i in range(len(img_list_ids))]
    qsts = json.load(open(dataset_path+qsts, 'r'))
    for q in qsts['questions']:
        im_id = q['image_id']
        qst_txt = q['question']
        qst_id = q['question_id']
        #pos_img = img_list_ids.index(im_id)
        pos_img = img_list_ids_idx[im_id]
        # insert qst info in qst_list
        qst_list[pos_img]['qst_ids'].append(qst_id)
        qst_list[pos_img]['qst_txt'].append(qst_txt)
    
    # reformat questions in single list
    qst_single_list = []
    qst_single_list_ids = []
    qst_single_list_ids_idx = dict()
    i = 0
    for q in qst_list:
        for id in q['qst_ids']:
            qst_single_list_ids.append(id)
            qst_single_list_ids_idx[id] = i
            i += 1
        for txt in q['qst_txt']:
            qst_single_list.append(txt)
    
    # store qst ids
    with open(dataset_path+out_qsts_ids, 'w') as f:
        for id in qst_single_list_ids:
            f.write(str(id)+'\n')
    # store qst txt
    with open(dataset_path+out_qsts, 'w') as f:
        for txt in qst_single_list:
            f.write(txt+'\n') 
    
    if(k in ANS_FILES.keys()):
        print 'Processing answers'
        # read answers and store in qst's order
        ans_list = [{'ans_ids': [], 'ans_txt': []} for i in range(len(qst_single_list))]
        anss = json.load(open(dataset_path+anss, 'r'))
        for a in anss['annotations']:
            qst_id = a['question_id']
            #pos_qst = qst_single_list_ids.index(qst_id)
            pos_qst = qst_single_list_ids_idx[qst_id]
            for a_ in a['answers']:
                ans_txt = a_['answer']
                ans_id = a_['answer_id']
                # insert ans info in ans_list
                ans_list[pos_qst]['ans_ids'].append(ans_id)
                ans_list[pos_qst]['ans_txt'].append(ans_txt)
        
        # apply preprocessing on answers?
        if data_preprocessing is not None:
            print "Preprocessing answers applying method "+data_preprocessing
            if data_preprocessing == 'single_frequent':
                k_freq = 1
            elif data_preprocessing == 'most_frequent':
                k_freq = k_frequent
                
            # preprocess each answer list
            for a_i, a in enumerate(ans_list):
                counts = Counter(a['ans_txt'])
                ans_list[a_i]['ans_txt'] = []
                sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))[::-1]
                for k_i in range(min(k_freq, len(sorted_counts))):
                    (ans,count) = sorted_counts[k_i]
                    ans_list[a_i]['ans_txt'].append(ans)
        
        # reformat answers in single list
        ans_single_list = []
        ans_single_list_ids = []
        for a in ans_list:
            for id in a['ans_ids']:
                ans_single_list_ids.append(id)
            for txt in a['ans_txt']:
                ans_single_list.append(txt)
        
        # store ans ids
        #with open(dataset_path+out_anss_ids, 'w') as f:
        #    for id in ans_single_list_ids:
        #        f.write(str(id)+'\n')
        # store ans txt
        with open(dataset_path+out_anss, 'w') as f:
            for txt in ans_single_list:
                f.write(txt.encode('utf-8')+'\n') 
        
print 'Done'
