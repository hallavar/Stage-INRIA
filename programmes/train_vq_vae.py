import numpy as np
import os
import glob
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Categorical

import time

from data_extraction import DataGenerator6


# cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number Gpus
ngpu = 1

# Number of training epochs
EPOCHS = 200

# Learning rate for optimizers
lr = 0.0002

out_dim = 256

decay = 0.0001

beta = 0.5

num_embeddings = 32

pth='/srv/tempdd/aschoen/preprocessed/'

img_shape = (28,28)

train_batch_size=2048
validation_batch_size=1024

labels = ['Benign','Malware']
limit=1000
limit=924170
split=4/5

train_dataset=DataGenerator6(pth, labels, limit, split, 'train', img_shape, 2, 16)
validation_dataset=DataGenerator6(pth, labels, limit, split, 'test', img_shape, 2, 16)

train_loader=DataLoader(train_dataset, train_batch_size, shuffle=True)
test_loader=DataLoader(validation_dataset, validation_batch_size, shuffle=True)

from models.vq_vae import VQVAE

vq_vae = VQVAE(1, 128, num_embeddings).to(device)
optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr, weight_decay=decay)

loss_recon = []
loss_quan = []

val_loss_recon = []
val_loss_quan = []

j = 0
count = 0
for epoch in range(EPOCHS):
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()

        x_recon, quantized, encoding_indices, vq_loss = vq_vae(x)


        recon_loss = F.mse_loss(x_recon, x)
        loss = vq_loss + recon_loss

        loss_recon.append(recon_loss.item())
        loss_quan.append(vq_loss.item())

        loss.backward()
        optimizer.step()
        
        if ((i+1) % 5 == 0):
            with torch.set_grad_enabled(False):
                x_val, _ = next(iter(test_loader))
                x_val = x_val.to(device)

                x_recon, quantized, encoding_indices, val_vq_loss = vq_vae(x)

                val_recon_loss = F.mse_loss(x_recon, x)

                val_loss = val_vq_loss + val_recon_loss

                val_loss_recon.append(val_recon_loss.item())
                val_loss_quan.append(val_vq_loss.item())

                print('[%d/%d][%d/%d]\t loss: %.4f\t val_loss: %.4f\t'
                          % (epoch, EPOCHS, i, len(train_loader),
                             loss.item(), val_loss.item()))

                j += 1

torch.save(vq_vae.state_dict(),'{}_vq_vae.pt'.format('pkt'))

from models.pixel_cnn import GatedPixelCNN, convert_dataset

z_train_dataset = convert_dataset(vq_vae, train_dataset, train_batch_size, device)
z_test_dataset = convert_dataset(vq_vae, validation_dataset, validation_batch_size, device)

in_channels = vq_vae.num_embeddings
pixel_cnn = GatedPixelCNN(10, in_channels, 64, 512, 12)
pixel_cnn.to(device)
    
optimizer = optim.Adam(pixel_cnn.parameters(), lr=1e-3, weight_decay=decay)

dataloader = torch.utils.data.DataLoader(z_train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(z_train_dataset, batch_size=validation_batch_size, shuffle=True)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device).squeeze(dim=1)
        y = y.to(device)

        optimizer.zero_grad()
        out = pixel_cnn(x, y)
        scores = out.permute(0, 2, 3, 1).reshape(-1, in_channels)

        loss = F.cross_entropy(scores, x.view(-1))

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if ((i+1) % 5 == 0):
            with torch.set_grad_enabled(False):
                x, y = next(iter(val_dataloader))
                x = x.to(device).squeeze(dim=1)
                y = y.to(device)

                out = pixel_cnn(x, y)
                scores = out.permute(0, 2, 3, 1).reshape(-1, in_channels)
                val_loss = F.cross_entropy(scores, x.view(-1))

                val_losses.append(val_loss.item())

                print('[%d/%d][%d/%d]\t loss: %.4f\t val_loss: %.4f\t'
                          % (epoch, EPOCHS, i, len(train_loader),
                             loss.item(), val_loss.item()))

class PixelCNNVQVAE(nn.Module):
    
    def __init__(self, pixelcnn, vqvae, latent_height, latent_width):
        super().__init__()
        self.pixelcnn = pixelcnn
        self.vqvae = vqvae
        self.latent_height = latent_height
        self.latent_width = latent_width
    
    def sample_prior(self, batch_size, label=None):
        indices = self.pixelcnn.sample(batch_size, (self.latent_height, self.latent_width), label).squeeze(dim=1)
        
        self.vqvae.eval()
        with torch.no_grad():
            quantized = self.vqvae.embed(indices).permute(0, 3, 1, 2)
        
        return quantized, indices
    
    def sample(self, batch_size, label=None):
        quantized, indices = self.sample_prior(batch_size, label)
        with torch.no_grad():
            x_recon = self.vqvae.decode(quantized)
        return x_recon, quantized, indices
    
    def generate(self, x):
        return self.vqvae(x)[0]
        
model = PixelCNNVQVAE(pixel_cnn, vq_vae, 7, 7)

torch.save(pixel_cnn.state_dict(), '{}_pixel_cnn.pt'.format("pkt"))


        

