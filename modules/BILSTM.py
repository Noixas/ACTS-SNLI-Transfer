
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from ModelBase import ModelBase
class BILSTM(ModelBase):
	def __init__(self):
		super().__init__()

		self.bilstm=nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
							bidirectional = True)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		embedding = lstm_out[:,-1]
		return embedding
		