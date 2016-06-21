from keras.engine import Input
from keras.engine.topology import merge
from keras.layers.core import Dropout, RepeatVector, Merge, Dense, Flatten, Activation, TimeDistributedDense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, Bidirectional
from keras.models import model_from_json, Sequential, Graph, Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from keras_wrapper.cnn_model import CNN_Model

import numpy as np
import cPickle as pk
import os
import logging
import shutil
import time


class VQA_Model(CNN_Model):
    
    def __init__(self, params, type='Basic_VQA_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None):
        """
            VQA_Model object constructor. 
            
            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param vocabularies: vocabularies used for GLOVE word embedding
            :param store_path: path to the folder where the temporal model packups will be stored

            References:
                [PReLU]
                Kaiming He et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

                [BatchNormalization]
                Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        """
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)
        
        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']
        
        self.verbose = verbose
        self._model_type = type
        self.params = params
        self.vocabularies = vocabularies

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path)

        # Prepare GLOVE embedding
        if params['GLOVE_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file "+ params['GLOVE_VECTORS'] +" >>>")
            self.word_vectors = np.load(os.path.join(params['GLOVE_VECTORS'])).item()
        else:
            self.word_vectors = dict()

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" VQA_Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('VQA_Model type "'+ type +'" is not implemented.')
        
        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path)
        
        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()
        
        
    def setOptimizer(self):

        """
            Sets a new optimizer for the Translation_Model.
        """

        # compile differently depending if our model is 'Sequential' or 'Graph'
        if self.verbose > 0:
            logging.info("Preparing optimizer and compiling.")

        if self.params['OPTIMIZER'] == 'adam':
            optimizer = Adam(lr=self.params['LR'])
        else:
            logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
            optimizer = self.params['OPTIMIZER']
        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'])

    
    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the Translation_Model instance.
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path


        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    def sampling(self, scores, sampling_type='max_likelihood', temperature=1.0):
        """
        Sampling words (each sample is drawn from a categorical distribution).
        Or picks up words that maximize the likelihood.
        In:
            scores - array of size #samples x #classes; 
                every entry determines a score for sample i having class j
            temperature - temperature for the predictions;
                the higher the flatter probabilities and hence more random answers

        Out:
            set of indices chosen as output, a vector of size #samples
        """
        if isinstance(scores, dict):
            scores = scores['output']
        if sampling_type == 'multinomial':
            logscores = np.log(scores) / temperature

            # numerically stable version
            normalized_logscores= logscores - np.max(logscores, axis=-1)[:, np.newaxis]
            margin_logscores = np.sum(np.exp(normalized_logscores),axis=-1)
            probs = np.exp(normalized_logscores) / margin_logscores[:, np.newaxis]

            #probs = probs.astype('float32')
            draws = np.zeros_like(probs)
            num_samples = probs.shape[0]
            # we use 1 trial to mimic categorical distributions using multinomial
            for k in xrange(num_samples):
                draws[k,:] = np.random.multinomial(1,probs[k,:],1)
            return np.argmax(draws, axis=-1)
        elif sampling_type == 'max_likelihood':
            return np.argmax(scores, axis=-1)
        else:
            raise NotImplementedError()

    def decode_predictions(self, preds, temperature, index2word, sampling_type, verbose=0):
        """
        Decodes predictions 
        
        In:
            preds - predictions codified as the output of a softmax activiation function
            temperature - temperature for sampling
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        # preds is a matrix of size #samples x #words
        answer_pred = map(lambda x:index2word[x], self.sampling(scores=preds, sampling_type=sampling_type, temperature=temperature))
        return answer_pred
    
    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for visualization
    # ------------------------------------------------------- #
    
    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'
        
        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'
            
        obj_str += '\n'
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'
            
        obj_str += '-----------------------------------------------------------------------------------'
        
        return obj_str
    
    
    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #


    def MultiEmbedding_VQA_Model_FusionLast(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Question model
        question = Input(name=self.ids_inputs[0], shape=tuple([params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        text_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                              input_length=params['MAX_INPUT_TEXT_LEN'], mask_zero=True, name='text_embedding')(question)
        lstm_text = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], name='lstm', return_sequences=False)(text_embedding)

        # Image model
        image = Input(name=self.ids_inputs[1], shape=tuple([params['IMG_FEAT_SIZE']]))
        image_embedding = Dense(params['IMG_EMBEDDING_HIDDEN_SIZE'], name='image_embedding')(image)
        if params['USE_BATCH_NORMALIZATION']:
            image_embedding = BatchNormalization(name='batch_normalization_image_embedding')(image_embedding)
        if params['USE_PRELU']:
            image_embedding = PReLU()(image_embedding)

        # Multimodal model
        image_text = merge([lstm_text, image_embedding], mode=params['MULTIMODAL_MERGE_MODE'])
        if params['USE_DROPOUT']:
            image_text = Dropout(0.5)(image_text)

        # Classifier
        classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                           activation=params['CLASSIFIER_ACTIVATION'])(image_text)

        self.model = Model(input=[question, image], output=classifier)



    def MultiEmbedding_Glove_VQA_Model_FusionLast(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors[word]
        self.word_vectors = {}

        # Question model
        question = Input(name=self.ids_inputs[0], shape=tuple([params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        text_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                                   input_length=params['MAX_INPUT_TEXT_LEN'],weights=[embedding_weights],
                                   trainable=params['GLOVE_VECTORS_TRAINABLE'],
                                   mask_zero=True, name='text_embedding')(question)
        lstm_text = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], name='lstm', return_sequences=False)(text_embedding)

        # Image model
        image = Input(name=self.ids_inputs[1], shape=tuple([params['IMG_FEAT_SIZE']]))
        image_embedding = Dense(params['IMG_EMBEDDING_HIDDEN_SIZE'], name='image_embedding')(image)
        if params['USE_BATCH_NORMALIZATION']:
            image_embedding = BatchNormalization(name='batch_normalization_image_embedding')(image_embedding)
        if params['USE_PRELU']:
            image_embedding = PReLU()(image_embedding)

        # Multimodal model
        image_text = merge([lstm_text, image_embedding], mode=params['MULTIMODAL_MERGE_MODE'])
        if params['USE_DROPOUT']:
            image_text = Dropout(0.5)(image_text)

        # Classifier
        classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                           activation=params['CLASSIFIER_ACTIVATION'])(image_text)

        self.model = Model(input=[question, image], output=classifier)


    def MultiEmbedding_Glove_Bidirectional_VQA_Model_FusionLast(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors[word]
        self.word_vectors = {}

        # Question model
        question = Input(name=self.ids_inputs[0], shape=tuple([params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        text_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                                   input_length=params['MAX_INPUT_TEXT_LEN'],weights=[embedding_weights],
                                   trainable=params['GLOVE_VECTORS_TRAINABLE'],
                                   mask_zero=True, name='text_embedding')(question)
        lstm_forward = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], 
                            name='forward', return_sequences=False)(text_embedding)
        lstm_backward = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], 
                            name='backward', go_backwards=True, return_sequences=False)(text_embedding)
        lstm_text = merge([lstm_forward, lstm_backward], mode='concat')

        # Image model
        image = Input(name=self.ids_inputs[1], shape=tuple([params['IMG_FEAT_SIZE']]))
        image_embedding = Dense(params['IMG_EMBEDDING_HIDDEN_SIZE'], name='image_embedding')(image)
        if params['USE_BATCH_NORMALIZATION']:
            image_embedding = BatchNormalization(name='batch_normalization_image_embedding')(image_embedding)
        if params['USE_PRELU']:
            image_embedding = PReLU()(image_embedding)

        # Multimodal model
        image_text = merge([lstm_text, image_embedding], mode=params['MULTIMODAL_MERGE_MODE'])
        if params['USE_DROPOUT']:
            image_text = Dropout(0.5)(image_text)

        # Classifier
        classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                           activation=params['CLASSIFIER_ACTIVATION'])(image_text)

        self.model = Model(input=[question, image], output=classifier)


    def MultiEmbedding_Glove_Bidirectional_DeepSoftmax_VQA_Model_FusionLast(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors[word]
        self.word_vectors = {}

        # Question model
        question = Input(name=self.ids_inputs[0], shape=tuple([params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        text_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                                   input_length=params['MAX_INPUT_TEXT_LEN'],weights=[embedding_weights],
                                   trainable=params['GLOVE_VECTORS_TRAINABLE'],
                                   mask_zero=True, name='text_embedding')(question)
        lstm_forward = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], name='forward', return_sequences=False)(text_embedding)
        lstm_backward = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], name='backward', go_backwards=True, return_sequences=False)(text_embedding)
        lstm_text = merge([lstm_forward, lstm_backward], mode='concat')

        # Image model
        image = Input(name=self.ids_inputs[1], shape=tuple([params['IMG_FEAT_SIZE']]))
        image_embedding = Dense(params['IMG_EMBEDDING_HIDDEN_SIZE'], name='image_embedding')(image)
        if params['USE_BATCH_NORMALIZATION']:
            image_embedding = BatchNormalization(name='batch_normalization_image_embedding')(image_embedding)
        if params['USE_PRELU']:
            image_embedding = PReLU()(image_embedding)

        # Multimodal model
        image_text = merge([lstm_text, image_embedding], mode=params['MULTIMODAL_MERGE_MODE'])
        if params['USE_DROPOUT']:
            image_text = Dropout(0.5)(image_text)

        for n_layer, size in enumerate(params['DEEP_SOFTMAX_LAYERS_SIZE']):
            if n_layer==0:
                fc = Dense(size, name='fc_'+str(n_layer))(image_text)
            else:
                fc = Dense(size, name='fc_'+str(n_layer))(fc)
            if params['USE_BATCH_NORMALIZATION']:
                fc = BatchNormalization()(fc)
            if params['USE_PRELU']:
                fc = PReLU()(fc)
            if params['USE_DROPOUT']:
                fc = Dropout(0.5)(fc)

        # Softmax classifier
        if len(params['DEEP_SOFTMAX_LAYERS_SIZE']) > 0: # deep MLP
            classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                           activation=params['CLASSIFIER_ACTIVATION'])(fc)

        else:

            classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                               activation=params['CLASSIFIER_ACTIVATION'])(image_text)
        # Classifier
        self.model = Model(input=[question, image], output=classifier)

 
    def MultiEmbedding_Glove_VQA_Model_LSTMAfterFusion(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors[word]
        self.word_vectors = {}

        # Question model
        question = Input(name=self.ids_inputs[0], shape=tuple([params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        text_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                                   input_length=params['MAX_INPUT_TEXT_LEN'],weights=[embedding_weights],
                                   trainable=params['GLOVE_VECTORS_TRAINABLE'],
                                   mask_zero=True, name='text_embedding')(question)
        lstm_text = LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'], name='lstm', return_sequences=False)(text_embedding)

        # Image model
        image = Input(name=self.ids_inputs[1], shape=tuple([params['IMG_FEAT_SIZE']]))
        image_embedding = Dense(params['IMG_EMBEDDING_HIDDEN_SIZE'], name='image_embedding')(image)
        if params['USE_BATCH_NORMALIZATION']:
            image_embedding = BatchNormalization(name='batch_normalization_image_embedding')(image_embedding)
        if params['USE_PRELU']:
            image_embedding = PReLU()(image_embedding)

        # Multimodal model
        image_text = merge([lstm_text, image_embedding], mode=params['MULTIMODAL_MERGE_MODE'])
        image_text = LSTM(params['LSTM_DECODER_HIDDEN_SIZE'], return_sequences=False, name='lstm_decoder')(image_text)
        if params['USE_DROPOUT']:
            image_text = Dropout(0.5)(image_text)

        # Classifier
        classifier = Dense(params['OUTPUT_VOCABULARY_SIZE'], name=self.ids_outputs[0],
                           activation=params['CLASSIFIER_ACTIVATION'])(image_text)

        self.model = Model(input=[question, image], output=classifier)


    # ------------------------------------------------------- #
    #       SAVE/LOAD
    #           Auxiliary methods for saving and loading the model.
    # ------------------------------------------------------- #
            
    def __getstate__(self): 
        """
            Behavour applied when pickling a VQA_Model instance.
        """ 
        obj_dict = self.__dict__.copy()
        del obj_dict['model']
        return obj_dict

