
from __future__ import absolute_import, division, unicode_literals
# from modules.LSTM import LSTMR



from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from modules.AverageEmbeddings import AverageEmbeddings
from torchtext.legacy.datasets.nli import SNLI
from torchtext.legacy.data import Field
from torchtext.legacy import data
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from modules.Classifier import Classifier
import torchtext
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import loggers
import os
import time
from datetime import timedelta
from pytorch_lightning.utilities import seed

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

import sys
import os
from typing import Text
import torch
import logging
from torchtext.legacy.data import Field
import torchtext
import torch.utils.data.dataset 
# get models.py from InferSent repo
# from models import InferSent

# Set PATHs
PATH_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data'
PATH_TO_W2V = '.vector_cache/glove.840B.300d.txt' #Modified Rodrigo
MODEL_PATH = 'trained_models/lstm/elated-waterfall-212/lstm.ckpt'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    # print(type(params))
    # print(type(samples))
    TEXT = Field(lower=True, include_lengths=True, batch_first=True,
                 tokenize='spacy', tokenizer_language="en_core_web_sm")
    
    # data = [' '.join(s) for s in samples],
    data = samples
    # print("data",len(data[0]))
    # print(data)
    TEXT.build_vocab(data, vectors=params.glove)
    
    params.model.emb_vec =  torch.nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True).to(device=params.device) 
    params["TEXT"] = TEXT
    #TODO: Pass to model
    #stoi is dict of word -> glove_id 
    # print(list(TEXT.vocab.stoi.keys())[:50])
    # print(TEXT.vocab['<unk>'])   # params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

#TODO:
## 

def batcher(params, batch):
    batch = [[" "] if len(sen) == 0 else sen for sen in batch] #Avoid empty sentences
    input = params.TEXT.process(batch) #Prepare data     
    input = [a.to(params.device) for a in input] #Send to device
    embeddings = params.model.encode_senval(input) #Encode with special fucntion for this
    return embeddings.detach().cpu().numpy()



"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""


def run_seneval(args):
    # Load InferSent model
    # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
    #                 'pool_type': 'max', 'dpout_model': 0.0, 'version': V}


    # Set params for SentEval
    if args.prototype:
        #Fast params
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}
    else:
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                        'tenacity': 5, 'epoch_size': 4}
    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)




    model = Classifier()

    model = model.load_from_checkpoint(args.checkpoint_path, model_name=args.model,disable_nonlinear=True)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # model.set_w2v_path(PATH_TO_W2V)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params_senteval["device"] =device
    params_senteval['model'] = model.cuda()
    params_senteval['glove'] = torchtext.vocab.GloVe(cache='.vector_cache/')
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks=['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC','MRPC', 'SICKEntailment']
    # transfer_tasks=['TREC']

    results = se.eval(transfer_tasks)
    print(results)
    macro = compute_macro_score(results)
    micro = compute_micro_score(results)
    results['macro'] = macro
    results['micro'] = micro
    wandb.log(results)
    save_results(args,results)


def get_checkpoint_path(args):
    if args.checkpoint_path != '':
        return args.checkpoint_path
    model_name = args.model
    return'trained_models/'+model_name+'/gold/'+model_name+'.ckpt'

def save_results(args, results):
    import json
    proto = '-fast' if args.prototype else '-full'
    path = args.save_results + args.model +proto + '.json'
    with open(path, 'w') as file:
     file.write(json.dumps(results)) # use `json.loads` to do the reverse
def compute_macro_score(results):
    total_dev_acc = 0
    len_metrics = len(results.keys())
    for k in results.keys():
        devacc = results[k]['devacc']
        total_dev_acc += devacc
    return total_dev_acc/len_metrics
def compute_micro_score(results):
    total_dev_acc = 0
    total_samples = 0
    len_metrics = len(results.keys())
    for k in results.keys():
        devacc = results[k]['devacc']*results[k]['ndev']
        total_samples += results[k]['ndev']
        total_dev_acc += devacc
    len_metrics*=total_samples
    return total_dev_acc/total_samples
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # encoders
    parser.add_argument("--model", default='awe',type=str)
    parser.add_argument("--save_results", default='results_senteval/',type=str)
    parser.add_argument("--checkpoint_path", default='',type=str)
    parser.add_argument('-p',"--prototype", action='store_true')
    parser.add_argument("--seed", default = 42, type=int)
    parser.add_argument("--batch", default = 64, type=int)

    args = parser.parse_args()
    wandb.init(project="atcs-seneval", config=args)
    wandb.log({"model_name":args.model})
    seed.seed_everything(args.seed)
    args.checkpoint_path = get_checkpoint_path(args)
    run_seneval(args)
    




