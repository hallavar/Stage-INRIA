# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:26:39 2021

@author: Utilisateur
"""
import os
import re
import glob
import random
#import cv2
import numpy as np
import tensorflow.keras as keras
from scapy.all import PcapReader #Maybe you will have to use PcapNgReader instead
#from boltons import iterutils
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
    l_labels=os.listdir(path) #Get the list of all the available label in the dataset
    label=[1 if cl in pcap.filename else 0 for cl in l_labels] #Create an array with 1 in the index of the corresponding label and 0 everywhere else
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

def convert_image_to_sequ(img,d=2,w=16):
    new_image=np.zeros((int(np.shape(img)[0]/d),int(np.shape(img)[1]/d)))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if i % d == 0 and j % d == 0:
                new_image[int(i/d), int(j/d)]=img[i,j]
    l=new_image.flatten()
    l=l//w
    u=np.asarray([l[2*p]*w+l[2*p+1] for p in range(int(len(l)/2))])
    u=u.astype('int32')
    u = [format(i,'02x') for i in u]
    return u
        

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
 
    # def __getitem__(self, index):
    #     samples=self.remnant[0]
    #     labels=self.remnant[1]
    #     while len(samples)<self.batch_size:
    #         pcap=random.choice(self.pcap_list)
    #         pkt=pcap.read_packet()
    #         label=get_label(pcap, os.path.commonpath(self.file_list))
    #         li=convert_pkt_to_bytes_sequ(pkt, self.suppr)
    #         li=iterutils.chunked(li, 0.5*np.prod(self.shape)/self.d**2)
    #         li=li[:(self.batch_size-len(samples))]
    #         last=li[len(li)-1]
    #         for i in range(0, len(li)):
    #             img=convert_sequ_to_image(li[i], self.d, self.w)
    #             samples.append(img)
    #             labels.append(label)
    #         if len(last) < 0.5*np.prod(self.shape)/self.d**2:
    #             # new_pad=np.sqrt(0.5*np.prod(self.shape)/len(last))
    #             # img=convert_sequ_to_image(last, round(new_pad), self.w)
    #             img = cv2.resize(img, self.shape)
    #             samples[len(samples)-1] = img
    
    def __getitem__(self, index):
        samples=[] #initialise the list of labels
        labels=[] #initialise the list of labels
        while len(samples)<self.batch_size: #We do this batch_size time
            pcap=random.choice(self.pcap_list) #We choose a random pcap to draw a packet
            pkt=pcap.read_packet() #We pick a packet from the randomly selected pcap
            label=get_label(pcap, os.path.commonpath(self.file_list)) #We get the label of the pcap
            li=convert_pkt_to_bytes_sequ(pkt, self.suppr) #We transform the label into a list of hexadecimals
            li=li[:np.prod(self.shape)/self.d**2] #We pick the n/d² first ones with n the size of the matrix
            li=li+['00']*(np.prod(self.shape)*self.d**2-len(li)) #padding
            img=convert_sequ_to_image(li, self.d, self.w) #create the image from the sequences
            samples.append(img)
            labels.append(label)
        return np.asarray(samples), np.asarray(labels) #get the batch
    
    def on_epoch_end(self):
        self.pcap_list=[PcapReader(file) for file in self.file_list if file.find('.lnk')==-1]
        
class DataGenerator2(keras.utils.Sequence):
    
    def __init__(self, path, shape, d, w, batch_size=5, shuffle=True):
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.labels = os.listdir(path)
        self.file_list=glob.glob(path+'/*/*.bin')
        self.shape = shape
        self.shuffle=shuffle
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))
 
    # def __getitem__(self, index):
    #     samples=self.remnant[0]
    #     labels=self.remnant[1]
    #     while len(samples)<self.batch_size:
    #         pcap=random.choice(self.pcap_list)
    #         pkt=pcap.read_packet()
    #         label=get_label(pcap, os.path.commonpath(self.file_list))
    #         li=convert_pkt_to_bytes_sequ(pkt, self.suppr)
    #         li=iterutils.chunked(li, 0.5*np.prod(self.shape)/self.d**2)
    #         li=li[:(self.batch_size-len(samples))]
    #         last=li[len(li)-1]
    #         for i in range(0, len(li)):
    #             img=convert_sequ_to_image(li[i], self.d, self.w)
    #             samples.append(img)
    #             labels.append(label)
    #         if len(last) < 0.5*np.prod(self.shape)/self.d**2:
    #             # new_pad=np.sqrt(0.5*np.prod(self.shape)/len(last))
    #             # img=convert_sequ_to_image(last, round(new_pad), self.w)
    #             img = cv2.resize(img, self.shape)
    #             samples[len(samples)-1] = img
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_file_temp = [self.file_list[k] for k in indexes]
        X, y = self.__data_generation(list_file_temp)
        return X, y
    
    def __data_generation(self, list_file_temp):
        labels=[]
        samples=[]
        ratio=int(np.prod(self.shape)/(2*self.d**2))
        for path in list_file_temp:
            label=[1 if cl in path else 0 for cl in self.labels]
            sample=open(path, 'rb').read()
            li=sample.hex()
            li=re.findall('..',li)
            li=li[:ratio] #We pick the n/d² first ones with n the size of the matrix
            li=li+['00']*(ratio-len(li)) #padding
            img=convert_sequ_to_image(li, self.d, self.w) #create the image from the sequences
            samples.append(img)
            labels.append(np.asarray(label))
        return np.asarray(samples), np.asarray(labels)
   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)