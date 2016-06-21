#from vqaEval import VQAEval
from vqaClassNormalizedEval import VQAClassNormalizedEval as VQAEval

# supported evaluators
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.vqa import vqaEval, visual_qa

from utils.read_write import list2vqa

import logging
from uuid import uuid4
import json
import os
import sys


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')



# COCO challenge metrics
def get_coco_score(gt_list, pred_list, verbose, extra_vars):
    """
    gt_list, dictionary of reference sentences (id, sentence)
    pred_list, dictionary of hypothesis sentences (id, sentence)
    verbose - if greater than 0 the metric measures are printed out
    extra_vars - extra variables, here are:
            extra_vars['language'] - the target language
    score, dictionary of scores

    """

    x_trgs = [x.lower() for x in gt_list]
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(pred_list)}
    refs = {idx: [rr] for idx, rr in enumerate(x_trgs)}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #(Meteor(language=extra_vars['language']),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores



# VQA challenge metric
def eval_vqa(gt_list, pred_list, verbose, extra_vars):
    quesFile = extra_vars['quesFile']
    annFile = extra_vars['annFile']
    
    # create temporal resFile
    resFile = 'tmp_res_file.json'
    list2vqa(resFile, pred_list, extra_vars['question_ids'])
    
    # create vqa object and vqaRes object
    vqa_ = visual_qa.VQA(annFile, quesFile)
    vqaRes = vqa_.loadRes(resFile, quesFile)
    vqaEval_ = vqaEval.VQAEval(vqa_, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    vqaEval_.evaluate()
    os.remove(resFile) # remove temporal file
    
    # get results
    acc_overall = vqaEval_.accuracy['overall']
    acc_yes_no = vqaEval_.accuracy['perAnswerType']['yes/no']
    acc_number = vqaEval_.accuracy['perAnswerType']['number']
    acc_other = vqaEval_.accuracy['perAnswerType']['other']
    #acc_per_class = vqaEval_.accuracy['perAnswerClass']
    #acc_class_normalized = vqaEval_.accuracy['classNormalizedOverall']

    if verbose > 0:
        logging.info('VQA Metric: Accuracy yes/no is {0}, other is {1}, number is {2}, overall is {3}'.\
                format(acc_yes_no, acc_other, acc_number, acc_overall))#, acc_class_normalized))
    return {'overall accuracy': acc_overall,
            'yes/no accuracy': acc_yes_no,
            'number accuracy': acc_number,
            'other accuracy': acc_other}
            #'class accuracy': acc_class_normalized,
            #'per answer class': acc_per_class}


########################################
# EVALUATION FUNCTIONS SELECTOR
########################################

# List of evaluation functions and their identifiers (will be used in params['METRICS'])
select = {
        'vqa': eval_vqa,
        'coco': get_coco_score
        }

                
                
########################################
# AUXILIARY FUNCTIONS
########################################

def vqa_store(question_id_list, answer_list, path):
    """
    Saves the answers on question_id_list in the VQA-like format.

    In:
        question_id_list - list of the question ids
        answer_list - list with the answers
        path - path where the file is saved
    """
    question_answer_pairs = []
    assert len(question_id_list) == len(answer_list), \
            'must be the same number of questions and answers'
    for q,a in zip(question_id_list, answer_list):
        question_answer_pairs.append({'question_id':q, 'answer':str(a)})
    with open(path,'w') as f:
        json.dump(question_answer_pairs, f)

def caption_store(samples, path):
    with open(path, 'w') as f:
            print >>f, '\n'.join(samples)

