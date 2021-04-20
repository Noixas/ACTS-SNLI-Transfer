
import torch

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import dropout
from torch.utils import data
from .AverageEmbeddings import AverageEmbeddings
from .LSTM import *
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import wandb
import time
from datetime import timedelta

# import numpy as np
# from pytorch_lightning.metrics import Accuracy, Precision, Recall
class Classifier(pl.LightningModule):
	def __init__(self,emb_vec=None, model_name='awe',input_dim=1200,hidden_dim=512, classes=3,disable_nonlinear=False):#input dim should be vectors.dim*3
		super().__init__()		
		if emb_vec != None:
			self.emb_vec = nn.Embedding.from_pretrained(emb_vec, freeze=True)	
		else:
			self.emb_vec = nn.Embedding(33635,300)
		self.emb_dim = 300#self.emb_vec.embedding_dim
		self.input_dim = 2048 *4 
		if model_name == 'awe':
			self.model = AverageEmbeddings()
			self.input_dim = 300 *4
		elif model_name=='lstm':
			self.model = LSTMR(self.emb_dim,2048) #2048 for lstm
		elif model_name=='bilstm':
			self.model = BILSTM(self.emb_dim,2048) #2048 for lstm
			self.input_dim = 2048 *2 *4 
		elif model_name=='bilstm-max':
			self.model = BILSTMMaxPool(self.emb_dim,2048) #2048 for lstm
			self.input_dim = 2048 *2 *4 
		#https://github.com/ihsgnef/InferSent-1/blob/6bea0ef38358a4b7f918cfacfc46d8607b516dc8/train_nli.py
		#Default droput is 0?
		self.dropout = 0
		self.hidden_dim = hidden_dim
		self.classes = classes
		if disable_nonlinear:
			self.classifier = nn.Sequential(
				nn.Linear(self.input_dim, self.hidden_dim ),
				nn.Linear(self.hidden_dim, self.hidden_dim ),
				nn.Linear(self.hidden_dim, self.classes))
		else:
			self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.classes),
                )
			
		self.loss = nn.CrossEntropyLoss()
		self.softmax= nn.Softmax(dim=1)

		self.start_time = time.time()
		##Metrics
		metrics = MetricCollection([Accuracy()])
		
		self.train_metrics = metrics.clone(prefix='train_')
		self.valid_metrics = metrics.clone(prefix='val_')
		self.train_acc = pl.metrics.Accuracy()
		self.valid_acc = pl.metrics.Accuracy()
		self.test_acc = pl.metrics.Accuracy()
		self.prev_val_acc = 0
		# embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

	def forward(self, x):
		premise, hypothesis, lengths_premise, lengths_hypothesis= x
		# print(premise.shape)
		# vect = self.emb_vec(premise)
		# print(vect.shape)
		# print('premise',premise.shape)
		# print('hypothesis',hypothesis.shape)
		u = self.model(premise,lengths_premise)
		v = self.model(hypothesis,lengths_hypothesis)
		diff = torch.abs(u - v)
		mul = torch.mul(u,v)
		print(v.shape)

		concat_emb =  torch.cat((u, v, diff,mul), 1)
		# print('cponcat',concat_emb.shape)
		# print(concat_emb.shape)
		linear_class = self.classifier(concat_emb) 
		return linear_class
	def encode_senval(self, x):
		sentences, lengths = x
		emb = self.emb_vec(sentences) #get glvoe emb
		emb = self.model(emb,lengths) #get lstm (and avriants) or awe emb
		return emb

	def configure_optimizers(self):
		optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
		lambda1 = lambda epoch: 0.99
		lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)
		return [optimizer], [lr_scheduler]

	def training_step(self, train_batch, batch_idx):

		x, lengths = train_batch
		premise,hypothesis,y =x 

		lengths_premise = premise[1]
		lengths_hypothesis = hypothesis[1]
		premise = premise[0]
		hypothesis = hypothesis[0]

		premise = self.emb_vec(premise)
		hypothesis = self.emb_vec(hypothesis)
		z = self.forward((premise,hypothesis,lengths_premise,lengths_hypothesis))    
		# print("ssssssssssssssssssssssssss\n",z)
		loss =self.loss(z, y)
		# print(loss)
		self.log('train_loss', loss)	
		soft_z = self.softmax(z)
		metrics = self.train_metrics(soft_z, y)
		self.train_acc(soft_z, y)
		# self.log('train_acc',self.train_acc(self.softmax(z), y), on_epoch=True)
		# wandb.log(metrics)
		# self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
		return loss
	def demo_inference(self,prem, hyp):
		""" Predict the NLI label for the tuple premise, hypothesis. 
		Returns a tuple of label in int form and text.
		"""
		
		premise, lengths_premise = prem
		hypothesis, lengths_hypothesis = hyp
		

		premise = self.emb_vec(premise)
		hypothesis = self.emb_vec(hypothesis)
		
		u = self.model(premise,lengths_premise).squeeze()
		v = self.model(hypothesis,lengths_hypothesis).squeeze()
		diff = torch.abs(u - v)
		mul = torch.mul(u,v)
		
		# print('cponcat',concat_emb.shape)
		# print(mul.shape)
		concat_emb =  torch.cat((u, v, diff,mul), 0)
		# print('cponcat',concat_emb.shape)
		# print(concat_emb.shape)
		linear_class = self.classifier(concat_emb) 
		soft = nn.Softmax(dim=0)	
		soft_z = soft(linear_class)
		pred = torch.argmax(soft_z).item()
		labels = ['entailment', 'contradiction', 'neutral']
		confidence = soft_z[pred].item()
		return pred, labels[pred], confidence, soft_z

	def training_epoch_end(self, outs):
		train_acc = self.train_acc.compute()
		self.log('train_acc_epoch', train_acc, on_step=False, on_epoch=True)

		# val_acc = torch.mean(torcSh.stack(val_acc))
	def validation_step(self, val_batch, batch_idx):

		x, _ = val_batch
		premise,hypothesis,y =x 

		lengths_premise = premise[1]
		lengths_hypothesis = hypothesis[1]
		premise = premise[0]
		hypothesis = hypothesis[0]


		premise = self.emb_vec(premise)
		hypothesis = self.emb_vec(hypothesis)
		z = self.forward((premise,hypothesis,lengths_premise,lengths_hypothesis))    
 
		
		loss=self.loss(z, y)
		# metrics = self.valid_metrics (self.softmax(z), y)
		val_acc = self.valid_acc(self.softmax(z), y)

		self.log('val_loss', loss)
		self.log('val_acc', val_acc)
		return val_acc
		

	def validation_epoch_end(self,val_acc):
		# print(val_acc)
		print('\n\n')
		val_acc = torch.mean(torch.stack(val_acc)).item() 
		lambda_stop  = 0.00001
		if (self.prev_val_acc+lambda_stop) > val_acc:
			self.optimizers().param_groups[0]['lr'] /= 5		
			print("Validation acc didnt improve. Dividing lr by 5 to ",self.optimizers().param_groups[0]['lr'])

		if self.optimizers().param_groups[0]['lr'] < 0.00001:
			self.trainer.should_stop = True

		self.log('val_acc_epoch', val_acc, on_step=False, on_epoch=True)
		stop = time.time()
		time_training = str(timedelta(seconds= stop - self.start_time))

		print("BATCH-Previous val acc:",self.prev_val_acc,' New val acc',val_acc)
		print("Time since training started:",time_training,'\n')
		# self.log('train_time', time_training)

		self.prev_val_acc = val_acc

	def test_step(self, val_batch, batch_idx):
		x, y = val_batch
		premise,hypothesis,y =x 
		
		
		lengths_premise = premise[1]
		lengths_hypothesis = hypothesis[1]
		premise = premise[0]
		hypothesis = hypothesis[0]

		premise = self.emb_vec(premise)
		hypothesis = self.emb_vec(hypothesis)
		z = self.forward((premise,hypothesis,lengths_premise,lengths_hypothesis))   
		
		loss=self.loss(z, y)
		# metrics = self.valid_metrics (self.softmax(z), y)
		test_acc = self.valid_acc(self.softmax(z), y)

		self.log('test_loss', loss)
		self.log('test_acc', test_acc)
		return test_acc

	def test_end(self,test_acc):
		# print(val_acc)
		print('\n')
		test_acc = torch.mean(torch.stack(test_acc)).item() 
		print("Test Acc",test_acc)
		self.log('test_acc_poch', test_acc, on_step=False, on_epoch=True)