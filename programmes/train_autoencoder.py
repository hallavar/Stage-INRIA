# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:39:31 2021

@author: Utilisateur
"""

import os
import numpy as np
import glob
import sys

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


#from tqdm import tqdm

from data_extraction import DataGenerator6
from models.vq_vae import VQVAE

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())
print(Variable(torch.randn(3,2,1).float().cuda()))

#torch.multiprocessing.freeze_support()


path="/srv/tempdd/aschoen/preprocessed"
labels=['Benign', 'Malware']

img_shape = (28,28)
train_batch_size=5096
validation_batch_size=10192

nb_files=1089370
limit=
split=2/3
    
print('Intitialisation du trainin_generator')
training_dataset=DataGenerator6(path, labels, limit, split, 'train', img_shape, 2, 16)
print('Initialisation du validation_generator')
validation_dataset=DataGenerator6(path, labels, limit, split, 'test',img_shape, 2, 16)
training_generator=DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
validation_generator=DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True)
    
def training_step(self, batch, batch_size, len_training_set):
        real_img, labels = batch
        real_img, labels= real_img.squeeze(0), labels.squeeze(0)
        results = self.forward(real_img, labels = labels)
        train_loss = self.loss_function(*results,
                                                  M_N = batch_size/len_training_set,
                                                  )    
        return train_loss
    
def train(model, training_data, validation_data, optimizer, num_epochs):  
    print("debut de l'entrainement")
    validation_loss=0
    for epoch in range(num_epochs):
        #loop = tqdm(training_data)
        print('\n')
        train_loss=0
        model.train()
        # ====================training=====================
        for idx, data in enumerate(training_data):
            # ===================forward====================
            loss = model.training_step(data, len(data[0]), 1/len(training_data))
            train_loss+=loss['loss']
            # ===================backward====================
            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()
            print(f'Epoch: [ {epoch} / {num_epochs}] : Batch [ {idx} / {len(training_data)} ] < training_loss : {train_loss} | validation_loss : {validation_loss} >',end='\r')
            # ====================validation=====================
        validation_loss=0
        model.eval()
        for data in validation_data:
            # ===================forward=====================
            loss = model.training_step(data, len(data[0]), 1/len(validation_data))
            validation_loss+=loss['loss'].data.item()
    return model
    
embedding_dim=20
num_embeddings=5
hidden_dims=[32, 64, 128]
img_size=img_shape[0]
num_epochs = 100
learning_rate = 1e-3
    
VQVAE.training_step=training_step
    
    
model=VQVAE(in_channels = 1, embedding_dim=embedding_dim, hidden_dims=hidden_dims, num_embeddings=num_embeddings, img_size=img_size).cuda()
optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

if __name__=='__main__':    
    torch.multiprocessing.set_start_method('spawn')
    model.share_memory()
    processes = []
    for i in range(4):
        p=torch.multiprocessing.Process(target=train, args=(model, training_generator, validation_generator, optimizer, num_epochs))
        p.start()
        processes.append(p)

    for p in processes : p.join()
