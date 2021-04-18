
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from .ModelBase import ModelBase

class AverageEmbeddings(ModelBase):
	def __init__(self):
		super().__init__()
		# self.save_hyperparameters()

	def forward(self, x,lengths):
		sentence_embedding = torch.mean(x,1)
		# print('mean',sentence_embedding.shape)
		return sentence_embedding

	# def configure_optimizers(self):
	# 	optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
	# 	return optimizer