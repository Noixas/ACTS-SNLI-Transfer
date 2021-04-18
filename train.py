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

def setup_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(wandb.run.dir, 'checkpoints'),
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}-{val_acc_epoch:.2f}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return checkpoint_callback, lr_monitor


def prepare_data(args):
    TEXT = Field(lower=True, include_lengths=True, batch_first=True,
                 tokenize='spacy', tokenizer_language="en_core_web_sm")
    LABEL = Field(sequential=False)
    # make splits for data

    print("Creating splits")
    if args.subset:
        train, dev, test = SNLI.splits(TEXT, LABEL, root='./subdata')
    else:
        train, dev, test = SNLI.splits(TEXT, LABEL, root='./data')
    print("Loading GloVe")
    glove = torchtext.vocab.GloVe(name='840B', dim=300)
    print("Aligning GloVe vocab")
    TEXT.build_vocab(train, vectors=glove)
    LABEL.build_vocab(train, specials_first=False)
    n_vocab = len(TEXT.vocab.itos)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Creating BucketIterator")
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(args.batch, 256, 256), device=device,shuffle=False)
    return TEXT, train_iter, dev_iter, test_iter


def setup_loggers(args):
    wandb_logger = WandbLogger()
    tb_logger = loggers.TensorBoardLogger(os.path.join(wandb.run.dir, 'tensorboard'),)
    return wandb_logger, tb_logger

def print_time(start,stop):
    time_setup = str(timedelta(seconds=stop - start))
    print(f"Time to setup:", time_setup)



##################################################################
##################################################################
#################       Train model         ######################
##################################################################
##################################################################



def train_model(args=None):
    start = time.time()

    TEXT, train_iter, dev_iter, test_iter = prepare_data(args)
    wandb_logger, tb_logger = setup_loggers(args)
    checkpoint_callback, lr_monitor = setup_callbacks(args)

    model = Classifier(emb_vec=TEXT.vocab.vectors,model_name=args.model,disable_nonlinear=args.disable_nonlinear)
    wandb_logger.watch(model)
    # num_sanity_val_steps=1, #
    
    auto_select_gpus =  False if args.gpus == 0 else True
    trainer = Trainer(gpus=args.gpus, auto_select_gpus=auto_select_gpus, fast_dev_run=args.dev, 
                    logger=[wandb_logger, tb_logger], precision=args.precision, callbacks=[lr_monitor, checkpoint_callback])

    print_time(start,time.time())
    trainer.fit(model, train_iter, dev_iter)
    wandb.log({'time_training_min': (time.time() - start)/60})
    print("\nTest model")
    trainer.test(model,test_iter)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # encoders
    parser.add_argument("--model", default='awe',type=str)
    # parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("-p","--precision",default=32, type=int)
    parser.add_argument("--seed", default = 42, type=int)

    parser.add_argument("--batch", default = 64, type=int)
    parser.add_argument("--dev", action='store_true')

    parser.add_argument("--disable_nonlinear", action='store_true')
    parser.add_argument("--subset", action='store_true',
                        help="Load the subset of the data instead for dev purpose")

    args = parser.parse_args()
    wandb.init(project="atcs-practical", config=args)

    seed.seed_everything(args.seed)
    train_model(args)
