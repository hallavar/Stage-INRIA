# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:26:39 2021

@author: Utilisateur
"""
import os
import re
import glob
import random
import cv2
import numpy as np
import tensorflow.keras as keras
from scapy.all import PcapReader #Maybe you will have to use PcapNgReader instead
from boltons import iterutils
from utils import hexa_repartition, get_closest_factors

def batch_of_raw_pkt(file, batch_size=64, t='hex'):
    pcap=PcapReader(file)
    batch=[]
    for i in range(0,batch_size):
        pkt=pcap.read_packet()
        pkt=bytes(pkt).hex()
        if t=='bin':
            pkt=bin(int(pkt, 16))[2:].zfill(8)
        batch.append(pkt)
    return np.asarray(batch)

def convert_pkt_to_bytes_sequ(pkt, suppr=34):
    pkt=bytes(pkt).hex() #Converting the packet to hexadecimal
    li=re.findall('..',pkt) #Get a list of bytes
    return li[suppr:]

def get_label(pcap, path='E:\stageINRIA\dataset\cic-ids-2018\labelled'):
    l_labels=os.listdir(path)
    label=[1 if cl in pcap.filename else 0 for cl in l_labels]
    return np.asarray(label)

def convert_sequ_to_image(li,d=2,w=16):
    li=[hexa_repartition(h, w) for h in li] #Map the 4-bit sequences (half of a byte) of all the bytes to a integer
    li=[value for sublist in li for value in sublist] #Get a list of all the mapped 4-bit sequences
    sh=len(li) #This will define the shape of the matrix
    shape=(get_closest_factors(sh),int(sh/get_closest_factors(sh))) #Get the accurate shape of an image according to the number of bytes in the packet
    shape=tuple([d*i for i in shape]) #multiply the shape by d (number of repeat)
    img=np.ones(shape)*-1 #Create an image of the corresponding shape full of negative value
    count=0 #Value to map in the image
    for i in range(0,shape[0]): #For each line of the image
        j=0 #We start at column 0
        while np.min(img[i])<0: #While there is still negative value in the line
            img[i:i+d,j:j+d]=li[count]*np.ones((d,d)) #affect the value to the [j,j+d] case
            j+=d #j increment by d
            count+=1# we go to the next value to map
    return np.asarray(img)

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, path, shape, d, w, suppr=34, batch_size=5, n_classes=2):
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.file_list=glob.glob(path+'/*/*/pcap/*')[300:]
        self.shape = shape
        self.suppr = suppr
        self.remnant=[[],[]]
        self.on_epoch_end()
 
    def __getitem__(self, index):
        samples=self.remnant[0]
        labels=self.remnant[1]
        while len(samples)<self.batch_size:
            pcap=random.choice(self.pcap_list)
            pkt=pcap.read_packet()
            label=get_label(pcap, os.path.commonpath(self.file_list))
            li=convert_pkt_to_bytes_sequ(pkt, self.suppr)
            li=iterutils.chunked(li, 0.5*np.prod(self.shape)/self.d**2)
            li=li[:(self.batch_size-len(samples))]
            last=li[len(li)-1]
            for i in range(0, len(li)):
                img=convert_sequ_to_image(li[i], self.d, self.w)
                samples.append(img)
                labels.append(label)
            if len(last) < 0.5*np.prod(self.shape)/self.d**2:
                # new_pad=np.sqrt(0.5*np.prod(self.shape)/len(last))
                # img=convert_sequ_to_image(last, round(new_pad), self.w)
                img = cv2.resize(img, self.shape)
                samples[len(samples)-1] = img
                
        self.remnant[0]=samples[self.batch_size:]
        self.remnant[1]=labels[self.batch_size:]
        samples = samples[:self.batch_size]
        labels = labels[:self.batch_size]
        return np.asarray(samples), np.asarray(labels)
    
    def on_epoch_end(self):
        self.pcap_list=[PcapReader(file) for file in self.file_list if file.find('.lnk')==-1]  