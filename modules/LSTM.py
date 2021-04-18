
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from .ModelBase import ModelBase
from torch.nn.utils import rnn
# Quick access: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html


class LSTMR(pl.LightningModule):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=False)

    def forward(self, x, lengths):
        # pack the input for better efficiency (lengths need to be in CPU)
        packed = rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)
        embedding = torch.squeeze(h_n)  # (batch_size, hidden_size)
        # lstm_out, _ = self.lstm(x)
        # embedding = lstm_out[:, -1]
        # return embedding
        return embedding

        # def configure_optimizers(self):
        # 	optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # 	return optimizer


class BILSTM(pl.LightningModule):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_size=emb_dim,
                              hidden_size=hidden_dim,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, x, lengths):
        # print(x.shape)
        packed = rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bilstm_out, (h_n, c_n) = self.bilstm(packed)
        # print(h_n.shape)
        # print(c_n.shape)
        # print(bilstm_out.shape)
        # print(h_n[0])
        # embedding = torch.cat((h_n[0, :,], h_n[1, :,]), dim=0)
        # print(embedding.shape)

        embedding = torch.squeeze(
            torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=-1))  # finally right shape
        # print(embedding.shape)
        return embedding

        # def configure_optimizers(self):
        # 	optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # 	return optimizer


class BILSTMMaxPool(BILSTM):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__(emb_dim, hidden_dim)

    def forward(self, x, lengths):
        packed = rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bilstm_out, (h_n, c_n) = self.bilstm(packed)
        h, _ = rnn.pad_packed_sequence(bilstm_out, batch_first=True)
        # print('first',h)
        # print('idk',_)

        len_batch = x.shape[0]
        h = h.view(len_batch, -1, 2, self.hidden_dim)

        # print(h_n.shape)
        # print('a',h.shape)
        h = torch.cat([h[:, :, 0, :], h[:, :, 1, :]], dim=-1)
        # print(h.shape)
        # print(h[0])
        embedding, _ = torch.max(h, dim=1)  # Maxpooling
        return embedding

# class ModelBase(pl.LightningModule):
# 	def __init__(self):
# 		super().__init__()
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
