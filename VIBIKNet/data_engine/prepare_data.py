from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

from collections import Counter
import operator

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

def build_dataset(params):
    
    if params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)
        max_text_len = params['MAX_INPUT_TEXT_LEN']

        ##### INPUT DATA
        ### QUESTIONS
        ds.setInput(base_path+'/'+params['QST_FILES']['train'][0], 'train',
                   type='text', id=params['INPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], build_vocabulary=True, fill=params['FILL'],
                   max_text_len=params['MAX_INPUT_TEXT_LEN'], max_words=params['INPUT_VOCABULARY_SIZE'],
                   repeat_set=params['REPEAT_QST'])
        ds.setInput(base_path+'/'+params['QST_FILES']['val'][0], 'val',
                   type='text', id=params['INPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], fill=params['FILL'],
                   max_text_len=params['MAX_INPUT_TEXT_LEN'], max_words=params['INPUT_VOCABULARY_SIZE'],
                   repeat_set=params['REPEAT_QST'])
        ds.setInput(base_path+'/'+params['QST_FILES']['test'][0], 'test',
                   type='text', id=params['INPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], fill=params['FILL'],
                   max_text_len=params['MAX_INPUT_TEXT_LEN'], max_words=params['INPUT_VOCABULARY_SIZE'],
                   repeat_set=params['REPEAT_QST'])
        ### QUESTIONS' associated IDs
        ds.setInput(base_path+'/'+params['QST_FILES']['train'][1], 'train',
                   type='id', id=params['INPUTS_IDS_DATASET'][0]+'_ids',
                   repeat_set=params['REPEAT_QST'])
        ds.setInput(base_path+'/'+params['QST_FILES']['val'][1], 'val',
                   type='id', id=params['INPUTS_IDS_DATASET'][0]+'_ids',
                   repeat_set=params['REPEAT_QST'])
        ds.setInput(base_path+'/'+params['QST_FILES']['test'][1], 'test',
                   type='id', id=params['INPUTS_IDS_DATASET'][0]+'_ids',
                   repeat_set=params['REPEAT_QST'])
        
        ### IMAGES
        ds.setInput(base_path+'/'+params['IMG_FILES']['train'][0], 'train',
                   type='image-features', id=params['INPUTS_IDS_DATASET'][1],
                   feat_len=params['IMG_FEAT_SIZE'],
                   repeat_set=params['REPEAT_IMG'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['val'][0], 'val',
                   type='image-features', id=params['INPUTS_IDS_DATASET'][1],
                   feat_len=params['IMG_FEAT_SIZE'],
                   repeat_set=params['REPEAT_IMG'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['test'][0], 'test',
                   type='image-features', id=params['INPUTS_IDS_DATASET'][1],
                   feat_len=params['IMG_FEAT_SIZE'],
                   repeat_set=params['REPEAT_IMG'])
        ### IMAGES' associated IDs
        ds.setInput(base_path+'/'+params['IMG_FILES']['train'][1], 'train',
                   type='id', id=params['INPUTS_IDS_DATASET'][1]+'_ids',
                   repeat_set=params['REPEAT_IMG'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['val'][1], 'val',
                   type='id', id=params['INPUTS_IDS_DATASET'][1]+'_ids',
                   repeat_set=params['REPEAT_IMG'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['test'][1], 'test',
                   type='id', id=params['INPUTS_IDS_DATASET'][1]+'_ids',
                   repeat_set=params['REPEAT_IMG'])
        

        ##### OUTPUT DATA
        ### ANSWERS
        ds.setOutput(base_path+'/'+params['ANS_FILES']['train'][0], 'train',
                   type='text', id=params['OUTPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], build_vocabulary=True, fill=params['FILL'],
                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])
        ds.setOutput(base_path+'/'+params['ANS_FILES']['val'][0], 'val',
                   type='text', id=params['OUTPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], fill=params['FILL'],
                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])
        if 'test' in params['ANS_FILES']:
            ds.setOutput(base_path+'/'+params['ANS_FILES']['test'][0], 'test',
                       type='text', id=params['OUTPUTS_IDS_DATASET'][0],
                       tokenization=params['TOKENIZATION_METHOD'], fill=params['FILL'],
                       max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])

        
        # Load extra variables (we need the original path to questions and annotations for VQA evaluation)
        ds.extra_variables['train'] = dict()
        ds.extra_variables['val'] = dict()
        ds.extra_variables['test'] = dict()
        
        ds.extra_variables['train']['quesFile'] = base_path+'/'+params['QST_FILES']['train'][2]
        ds.extra_variables['val']['quesFile'] = base_path+'/'+params['QST_FILES']['val'][2]
        ds.extra_variables['test']['quesFile'] = base_path+'/'+params['QST_FILES']['test'][2]
        
        ds.extra_variables['train']['annFile'] = base_path+'/'+params['ANS_FILES']['train'][1]
        ds.extra_variables['val']['annFile'] = base_path+'/'+params['ANS_FILES']['val'][1]
        if 'test' in params['ANS_FILES']:
            ds.extra_variables['test']['annFile'] = base_path+'/'+params['ANS_FILES']['tes'][1]
        
        
        # Remove all samples of the train set not belonging to the top classes chosen
        if params['KEEP_TOP_ANSWERS']:
            ds.keepTopOutputs('train', params['OUTPUTS_IDS_DATASET'][0], params['OUTPUT_VOCABULARY_SIZE'])
        # Filter top K answers per question-image pair
        if params['FILTER_ANSWERS']:
            filter_k_frequent_answers(ds, params)
        
        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATA_ROOT_PATH'])
    
    
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATA_ROOT_PATH']+'/Dataset_'+params['DATASET_NAME']+'.pkl')

    return ds



def filter_k_frequent_answers(ds, params, filter_sets=['train', 'val']): # the test set does not have answers
    k_filter = params['K_FILTER']
    
    # Process each set split
    for s in filter_sets:
        
        logging.info('Filtering top '+str(k_filter)+' answers on set split "'+s+'" of the dataset '+ds.name)
        
        # Get question_ids and answers and store in dict([qst_id])
        ans_list = dict()
        exec('qst_ids = ds.X_'+s+'["'+params['INPUTS_IDS_DATASET'][0]+'_ids"]')
        exec('ans = ds.Y_'+s+'["'+params['OUTPUTS_IDS_DATASET'][0]+'"]')
        for i,(q,a) in enumerate(zip(qst_ids, ans)):
            try:
                ans_list[q]['ans_txt'].append(a)
                ans_list[q]['pos'].append(i)
            except:
                ans_list[q] = dict()
                ans_list[q]['ans_txt'] = []
                ans_list[q]['pos'] = []
                ans_list[q]['ans_txt'].append(a)
                ans_list[q]['pos'].append(i)
         
        # Count occurrences for each question separately and index the more frequent
        kept = []
        for key,a in ans_list.iteritems():
            counts = Counter(a['ans_txt'])
            sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
            for k_i in range(min(k_filter, len(sorted_counts))):
                (ans,count) = sorted_counts[k_i]
                pos_ans = ans_list[key]['ans_txt'].index(ans)
                kept.append(ans_list[key]['pos'][pos_ans])
    
        # Keep the most frequent answers only and remove the rest
        # Inputs
        exec('ids = ds.X_'+s+'.keys()')
        for id in ids:
            exec('ds.X_'+s+'[id] = [ds.X_'+s+'[id][k] for k in kept]')
        # Outputs
        exec('ids = ds.Y_'+s+'.keys()')
        for id in ids:
            exec('ds.Y_'+s+'[id] = [ds.Y_'+s+'[id][k] for k in kept]')
        new_len = len(kept)
        exec('ds.len_'+s+' = new_len')
        
        logging.info('Samples remaining in set split "'+s+'": '+str(new_len))
    
