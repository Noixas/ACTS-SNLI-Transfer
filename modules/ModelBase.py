
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class ModelBase(pl.LightningModule):
	def __init__(self):
		super().__init__()


	# def training_step(self, train_batch, batch_idx):
	# 	x, y = train_batch
	# 	x = x.view(x.size(0), -1)
	# 	z = self.encoder(x)    
	# 	x_hat = self.decoder(z)
	# 	loss = F.mse_loss(x_hat, x)
	# 	self.log('train_loss', loss)
	# 	return loss

	# def validation_step(self, val_batch, batch_idx):
	# 	x, y = val_batch
	# 	x = x.view(x.size(0), -1)
	# 	z = self.encoder(x)
	# 	x_hat = self.decoder(z)
	# 	loss = F.mse_loss(x_hat, x)
	# 	self.log('val_loss', loss)

