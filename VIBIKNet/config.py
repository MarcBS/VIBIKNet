def load_parameters():
    """
        Loads the defined parameters
    """

    # Input data params
    DATA_ROOT_PATH = '/home/lvapeab/smt/tasks/image_desc/VQA/'  # Root path to the data
    #DATA_ROOT_PATH = '/media/HDD_2TB/DATASETS/VQA/'
    DATASET_NAME = 'VQA'  # Dataset name

    IMG_FILES = {'train': ['Images/mscoco/train_list_features.txt',  # Image and features files
                           'Images/mscoco/train_list_ids.txt'],
                 'val': ['Images/mscoco/val_list_features.txt',
                         'Images/mscoco/val_list_ids.txt'],
                 'test': ['Images/mscoco/test_list_features.txt',
                          'Images/mscoco/test_list_ids.txt']
                }
    REPEAT_IMG = 10*3 # we have 3 questions per image
    QST_FILES = {'train': ['Questions/OpenEnded_mscoco_train2014_questions.txt',  # Question files
                           'Questions/OpenEnded_mscoco_train2014_questions_ids.txt',
                           'Questions/OpenEnded_mscoco_train2014_questions.json'],
                 'val': ['Questions/OpenEnded_mscoco_val2014_questions.txt',
                         'Questions/OpenEnded_mscoco_val2014_questions_ids.txt',
                         'Questions/OpenEnded_mscoco_val2014_questions.json'],
                 'test': ['Questions/OpenEnded_mscoco_test2015_questions.txt',
                          'Questions/OpenEnded_mscoco_test2015_questions_ids.txt',
                          'Questions/OpenEnded_mscoco_test2015_questions.json']
                }
    REPEAT_QST = 10 # we have 10 answers per question
    ANS_FILES = {'train': ['Annotations/mscoco_train2014_annotations.txt',  # Answer files
                           'Annotations/mscoco_train2014_annotations.json'],
                 'val': ['Annotations/mscoco_val2014_annotations.txt',
                         'Annotations/mscoco_val2014_annotations.json']
                 }

    # Dataset parameters
    INPUTS_IDS_DATASET = ['question', 'image']  # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['answer']  # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['question', 'image']  # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['answer']  # Corresponding outputs of the built model

    # Evaluation params
    METRICS = ['vqa']  # Metric used for evaluating model after each epoch
    EVAL_ON_SETS = ['val']  # Possible values: 'train', 'val' and 'test'
    START_EVAL_ON_EPOCH = 5  # First epoch where the model will be evaluated
    SAMPLING = 'max_likelihood'  # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1  # Multinomial sampling parameter

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_questions'  # Select which tokenization we'll apply:
                                                # 'tokenize_basic' or 'tokenize_questions' (recommended)
    FILL = 'end' # Select the padding mode: 'start' or 'end' (recommended)

    # Input image parameters
    IMG_FEAT_SIZE = 1024  # Size of the image features

    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0  # Size of the input vocabulary. Set to 0 for using all, otherwise will be truncated to these most frequent words.
    MAX_INPUT_TEXT_LEN = 35  # Maximum length of the input text

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 2000 # vocabulary of output text. Set to 0 for autosetting, otherwise will be truncated
    MAX_OUTPUT_TEXT_LEN = 0 # set to 0 if we want to use the whole answer as a single class
    KEEP_TOP_ANSWERS = True # select only top 'OUTPUT_VOCABULARY_SIZE' answers from the training set and remove remaining samples
    FILTER_ANSWERS = True # filter top appearing answers for a single question-image pair
    K_FILTER = 1 # number of answers per question kept after filtering

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'
    CLASS_MODE = 'categorical'
    LR_DECAY = 20  # number of minimum number of epochs before the next LR decay
    LR_GAMMA = 0.8  # multiplier used for decreasing the LR
    LR = 0.001  # 0.001 recommended for adam optimizer

    # Training parameters
    MAX_EPOCH = 20  # Stop when computed this number of epochs
    BATCH_SIZE = 4096
    PARALLEL_LOADERS = 10  # parallel data batch loaders
    EPOCHS_FOR_SAVE = 1  # number of epochs between model saves
    WRITE_VALID_SAMPLES = True  # Write valid samples in file

    # Model parameters
    # ===
    # Possible MODEL_TYPE values: 
    #                          [ Recommended Models ]
    #
    #                          MultiEmbedding_VQA_Model_FusionLast
    #                          MultiEmbedding_Glove_VQA_Model_FusionLast
    #                          MultiEmbedding_Glove_Bidirectional_VQA_Model_FusionLast
    #                          MultiEmbedding_Glove_DeepSoftmax_VQA_Model_FusionLast
    #                          MultiEmbedding_Glove_VQA_Model_LSTMAfterFusion
    #
    #                          [ NON-recommended Models ]
    #
    #                          Basic_VQA_Model, MultiEmbedding_VQA_Model, Bidirectional_VQA_Model, 
    #                          MultiEmbedding_Bidirectional_VQA_Model, MultiEmbedding_Bidirectional_DeepSoftmax_VQA_Model,
    #                          MultiEmbedding_Glove_Bidirectional_DeepSoftmax_VQA_Model,
    # ===
    MODEL_TYPE = 'MultiEmbedding_Glove_Bidirectional_VQA_Model_FusionLast'

    # Input text parameters
    GLOVE_VECTORS = DATA_ROOT_PATH + 'Glove/' + 'glove_300.npy'  # Path to pretrained vectors. Set to None if you don't want to use pretrained vectors.
    GLOVE_VECTORS_TRAINABLE = True  # Finetune or not the word embedding vectors.
    TEXT_EMBEDDING_HIDDEN_SIZE = 300  # When using pretrained word embeddings, this parameter must match with the word embeddings size

    # Layer dimensions
    LSTM_ENCODER_HIDDEN_SIZE = 250  # For models with LSTM encoder
    IMG_EMBEDDING_HIDDEN_SIZE = 500  # Visual embedding size
    MULTIMODAL_MERGE_MODE = 'sum'  # 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'
    LSTM_DECODER_HIDDEN_SIZE = 500  # for models with LSTM decoder
    DEEP_SOFTMAX_LAYERS_SIZE = []  # additional Fully-Connected layers's sizes applied before softmax
    USE_DROPOUT = True  # Use dropout (0.5)
    USE_BATCH_NORMALIZATION = False  # if True it is recommended to deactivate Dropout
    USE_PRELU = False  # use PReLU or use ReLU instead
    CLASSIFIER_ACTIVATION = 'softmax'

    # Results plot and models storing parameters
    MODEL_NAME = MODEL_TYPE + '_txtemb_' + str(TEXT_EMBEDDING_HIDDEN_SIZE) + \
                 '_igmemb_' + str(IMG_EMBEDDING_HIDDEN_SIZE) + \
                 '_' + MULTIMODAL_MERGE_MODE + '_lstm_' + str(LSTM_ENCODER_HIDDEN_SIZE)

    STORE_PATH = 'trained_models/' + MODEL_NAME  # models and evaluation results will be stored here
    SAMPLING_SAVE_MODE = 'vqa'  # 'list' or 'vqa'
    VERBOSE = 1  # Verbosity
    RELOAD = 0  # If 0 start training from scratch, otherwise the model saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True  # build again or use stored instance
    MODE = 'training'  # 'training' or 'sampling' (if 'sampling' then RELOAD must be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
