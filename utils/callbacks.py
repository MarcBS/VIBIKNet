from __future__ import print_function

"""
Extra set of callbacks.
"""

import random
import warnings
import numpy as np
import logging

from utils import evaluation
from utils.read_write import *

from keras.callbacks import Callback as KerasCallback

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

###
# Printing callbacks
###


class PrintPerformanceMetricOnEpochEnd(KerasCallback):

    def __init__(self, model, dataset, gt_id, metric_name, set_name, batch_size, extra_vars=dict(), 
                 is_text=False, index2word_y=None, sampling='max_likelihood',
                 write_samples = False, save_path='logs/performance.', reload_epoch=0,
                 start_eval_on_epoch=0, write_type='list', sampling_type='max_likelihood', verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param batch_size: batch size used during sampling
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param write_samples: flag for indicating if we want to write the predicted data in a text file
            :param save_path: path to dumb the logs
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
            :param write_type: method used for writing predictions
            :param sampling_type: type of sampling used (multinomial or max_likelihood)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.sampling = sampling
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.extra_vars = extra_vars
        self.save_path = save_path
        self.reload_epoch = reload_epoch
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1 # start by index 1
        epoch += self.reload_epoch
        if epoch < self.start_eval_on_epoch and self.verbose > 0:
            logging.info('Not evaluating until end of epoch '+ str(self.start_eval_on_epoch))
            return
        
        # Evaluate on each set separately
        for s in self.set_name:

            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size, 
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'], 'predict_on_sets': [s]}
            predictions = self.model_to_eval.predictNet(self.ds, params_prediction)[s]
            gt_y = eval('self.ds.Y_'+s+'["'+self.gt_id+'"]')
            
            if(self.is_text):
                # Convert predictions into sentences
                predictions = self.model_to_eval.decode_predictions(predictions, 1, # always set temperature to 1 
                                                      self.index2word_y, 
                                                      self.sampling_type, 
                                                      verbose=self.verbose)
            
            # Store predictions
            if self.write_samples:
                # Store result
                filepath = self.save_path +'/'+ s +'_epoch_'+ str(epoch) +'.pred' # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])

            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric '+metric)
                filepath = self.save_path +'/'+ s +'.'+metric # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                            gt_list=gt_y, 
                            pred_list=predictions, 
                            verbose=self.verbose,
                            extra_vars=self.extra_vars[s])

                # Print results to file
                with open(filepath, 'a') as f:
                    header = 'epoch,'
                    line = str(epoch)+','
                    for metric_, value in metrics.iteritems():
                        #if(not m['name'] == 'per answer class'):
                        header += metric_ +','
                        line += str(value)+','
                    if(epoch==1 or epoch==self.start_eval_on_epoch):
                        f.write(header+'\n')
                    f.write(line+'\n')
                if self.verbose > 0:
                    logging.info('Done evaluating on metric '+metric)
                            
###


class EarlyStopping(KerasCallback):
    """
    Reduces learning rate during the training.

    Original work: jiumem [https://github.com/jiumem]
    """
    def __init__(self, patience=0, reduce_nb=10, is_early_stopping=True, verbose=1):
        """
        In:
            patience - number of beginning epochs without reduction;
                by default 0
            reduce_rate - multiplicative rate reducer; by default 0.5
            reduce_nb - maximal number of reductions performed; by default 10
            is_early_stopping - if true then early stopping is applied when
                reduce_nb is reached; by default True
            verbose - verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.is_early_stopping = is_early_stopping
        self.verbose = verbose
        self.epsilon = 0.1e-10

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score is None:
            warnings.warn('validation score is off; ' +
                    'this reducer works only with the validation score on')
            return
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.is_early_stopping:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1
