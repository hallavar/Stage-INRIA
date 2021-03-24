# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:39:31 2021

@author: Utilisateur
"""

import os
import numpy as np
import glob

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from time import sleep



# from sklearn.model_selection import train_test_split

# from data_extraction import DataGenerator2
from data_extraction import DataGenerator5
from models.vq_vae import VQVAE
if __name__ == '__main__':
    #torch.multiprocessing.freeze_support()
    path="..\..\dataset\cic-ids-2018\preprocessed"
    labels=['Benign', 'Malware']
    #list_pcap=[os.path.join(path,name) for name in list_pcap]
    
    # list_label=glob.glob(path)
    # list_days=[]
    # list_pcap=[]
    # for i, label in enumerate(list_label):
    #     list_days.append(glob.glob(label))
    #     list_pcap.append([])
    #     for j, day in enumerate(list_days[i]):
    #         list_pcap[i].append(glob.glob(day))
    #     list_days[i]=dict(zip(range(len(list_days[i])),list_pcap[i]))
    # list_label=dict(zip(range(len(list_label)),list_days))
        
    img_shape = (28,28)
    train_batch_size=64
    validation_batch_size=512
    # data=glob.iglob(path+'/*/*/*/*.bin')
    #nb_files=84751439
    nb_files=1089370
    limit=1000
    split=2/3
    
    #training_data, validation_data = train_test_split(data)
    
    
    training_generator=DataGenerator5(path, labels, limit, split, 'train', img_shape, 2, 16)
    validation_generator=DataGenerator5(path, labels, limit, split, 'test',img_shape, 2, 16)
    training_generator=DataLoader(training_generator, batch_size=train_batch_size, shuffle=True, num_workers=0)
    validation_generator=DataLoader(validation_generator, batch_size=validation_batch_size, shuffle=True, num_workers=0)
    
    def training_step(self, batch, batch_size, len_training_set):
            real_img, labels = batch
            # real_img = torch.from_numpy(real_img/255).float()
            # real_img=real_img.unsqueeze(1)
            # real_img=Variable(real_img).cuda()
            real_img, labels= real_img.squeeze(0), labels.squeeze(0)
            results = self.forward(real_img, labels = labels)
            train_loss = self.loss_function(*results,
                                                  M_N = batch_size/len_training_set,
                                                  )
            
            return train_loss
    
    def train(model, training_data, validation_data, optimizer, num_epochs):  
        validation_loss=0
        for epoch in range(num_epochs):
            loop = tqdm(training_data)
            train_loss=0
            model.train()
            # ====================training=====================
            for idx, data in enumerate(loop):
                # ===================forward====================
                loss = model.training_step(data, len(data[0]), 1/len(training_data))
                train_loss+=loss['loss']
                # ===================backward====================
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss=train_loss.data.item(), validation_loss=validation_loss)
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
    
    
    
    model = train(model, training_generator, validation_generator, optimizer, num_epochs)