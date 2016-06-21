import logging
import numpy as np
from timeit import default_timer as timer
import copy

from keras_wrapper.cnn_model import saveModel, loadModel

from config import load_parameters
from vqa_model import VQA_Model
from data_engine.prepare_data import build_dataset
from utils.evaluation import vqa_store
from utils.callbacks import PrintPerformanceMetricOnEpochEnd
from utils.read_write import *

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def main(params):
    """
        Main function
    """

    if(params['RELOAD'] > 0):
        logging.info('Resuming training.')

    check_params(params)

    ########### Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    ###########


    ########### Build model
    if(params['RELOAD'] == 0): # build new model 
        vqa = VQA_Model(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                        model_name=params['MODEL_NAME'], vocabularies=dataset.vocabulary,
                        store_path=params['STORE_PATH'])
        
        # Define the inputs and outputs mapping from our Dataset instance to our model
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = vqa.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        vqa.setInputsMapping(inputMapping)
            
        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = vqa.ids_outputs[i]
            outputMapping[id_dest] = pos_target
        vqa.setOutputsMapping(outputMapping)
        
    else: # resume from previously trained model
        vqa = loadModel(params['STORE_PATH'], params['RELOAD'])
        vqa.setOptimizer()
    ###########

    
    ########### Callbacks
    callbacks = buildCallbacks(params, vqa, dataset)
    ###########
    

    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'], 'batch_size': params['BATCH_SIZE'], 
                       'lr_decay': params['LR_DECAY'], 'lr_gamma': params['LR_GAMMA'], 
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD']}
    vqa.trainNet(dataset, training_params)
    

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
    ###########
    

def apply_VQA_model(params):
    """
        Function for using a previously trained model for sampling.
    """
    
    ########### Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    ###########
    
    
    ########### Load model
    vqa = loadModel(params['STORE_PATH'], params['RELOAD'])
    vqa.setOptimizer()
    ###########
    

    ########### Apply sampling
    for s in params["EVAL_ON_SETS"]:

        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'], 'n_parallel_loaders': params['PARALLEL_LOADERS'], 'predict_on_sets': [s]}
        predictions = vqa.predictNet(dataset, params_prediction)[s]
            
        # Convert predictions into sentences
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        predictions = vqa.decode_predictions(predictions, 1, # always set temperature to 1 
                                                            vocab, params['SAMPLING'], verbose=params['VERBOSE'])
        
        # Store result
        filepath = vqa.model_path+'/'+ s +'_sampling.txt' # results file
        if params['SAMPLING_SAVE_MODE'] == 'list':
            list2file(filepath, predictions)
        elif params['SAMPLING_SAVE_MODE'] == 'vqa':
            exec('question_ids = dataset.X_'+s+'["'+params['INPUTS_IDS_DATASET'][0]+'_ids"]')
            list2vqa(filepath, predictions, question_ids)
    ###########
    


def buildCallbacks(params, model, dataset):
    """
        Builds the selected set of callbacks run during the training of the model
    """
    
    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = dict()
        extra_vars['n_parallel_loaders'] = params['PARALLEL_LOADERS']
        for s in params['EVAL_ON_SETS']:
            extra_vars[s] = dict()
            exec('extra_vars[s]["question_ids"] = dataset.X_'+s+'["'+params['INPUTS_IDS_DATASET'][0]+'_ids"]')
            extra_vars[s]['quesFile'] = dataset.extra_variables[s]['quesFile']
            extra_vars[s]['annFile'] = dataset.extra_variables[s]['annFile']
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        callback_metric = PrintPerformanceMetricOnEpochEnd(model, dataset, gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                               metric_name=params['METRICS'], set_name=params['EVAL_ON_SETS'],
                                                               batch_size=params['BATCH_SIZE'],
                                                               is_text=True, index2word_y=vocab, # text info
                                                               sampling_type=params['SAMPLING'], # text info
                                                               save_path=model.model_path,
                                                               reload_epoch=params['RELOAD'],
                                                               start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                                               write_samples=True,
                                                               write_type=params['SAMPLING_SAVE_MODE'],
                                                               extra_vars=extra_vars,
                                                               verbose=params['VERBOSE'])
        callbacks.append(callback_metric)

    
    
    return callbacks



def check_params(params):
    if 'Glove' in params['MODEL_TYPE'] and params['GLOVE_VECTORS'] is None:
        logger.warning("You set a model that uses pretrained word vectors but you didn't specify a vector file."
                       "We'll train WITHOUT pretrained embeddings!")
    if params["USE_DROPOUT"] and params["USE_BATCH_NORMALIZATION"]:
        logger.warning("It's not recommended to use both dropout and batch normalization")
    if params['MODE'] == 'sampling':
        assert len(params["EVAL_ON_SETS"]) == 1, 'It is only possible to sample over 1 set'
    if 'Bidirectional' in params["MODEL_TYPE"]:
        assert params["LSTM_ENCODER_HIDDEN_SIZE"]*2 == params["IMG_EMBEDDING_HIDDEN_SIZE"], "LSTM_ENCODER_HIDDEN_SIZE must be IMG_EMBEDDING_HIDDEN_SIZE/2"



if __name__ == "__main__":
 
    params = load_parameters()
    check_params(params)
    if(params['MODE'] == 'training'):
        logging.info('Running training.')
        main(params)
    elif(params['MODE'] == 'sampling'):
        logging.info('Running sampling.')
        apply_VQA_model(params)

    logging.info('Done!')   



