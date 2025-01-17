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
import torch
#from tensorflow import keras
#from scapy.all import PcapReader #Maybe you will have to use PcapNgReader instead
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
        
    
class DataGenerator2():
    
    def __init__(self, paths, shape, d, w, batch_size=5, shuffle=True):
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.labels=os.listdir(os.path.dirname(paths[0]))
        self.file_list=paths
        self.shape = shape
        self.shuffle=shuffle
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))
 
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

class DataGenerator3():
    
    def __init__(self, paths_dict, N, shape, d, w, batch_size=5, shuffle=True):
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.labels=os.listdir(os.path.dirname(paths[0]))
        self.file_dict=paths
        self.N=N
        self.shape = shape
        self.shuffle=shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(N/self.batch_size))
         
    def __getitem__(self, index):
        list_file_temp=[]
        for i in self.batch_size:
            lbl=random.choice(self.file_dict.values)
            day=random.choice(lbl.values)
            pcap=random.choice(day)
            n=int(N/8000) # 8000 : aproximately the number of directory in a given day
            rdm = random.randint(0, n)
            name=pcap+'\\'+pcap.split('\\')[-1]+'_'+str(rdm)+'.bin'
            while os.path.isfile(name)==False:
                rdm = random.randint(0, n)
                name=pcap+'\\'+pcap.split('\\')[-1]+'_'+str(rdm)+'.bin'
            list_file_temp.append(name)
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
    
class DataGenerator4(torch.utils.data.Dataset):
    
    def __init__(self, path, N, limit, shape, d, w, batch_size=5, shuffle=True):
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.labels=['Benign']
        self.list_pcap=[os.path.join(path,name) for name in os.listdir(path)][:limit]
        self.N=N
        self.shape = shape
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last=os.path.join(self.list_pcap[0],os.listdir(self.list_pcap[0])[0])
    
    def __len__(self):
        return int(np.ceil(self.N/self.batch_size))
         
    def __getitem__(self, index):
        list_file_temp=[]
        folder, name = os.path.split(self.last)
        name, nbr=name.split('.bin')[0].split('_')
        for i in range(self.batch_size):
            nbr = str(int(nbr)+1)
            temp_path=os.path.join(folder,name+'_'+nbr+'.bin')
            if os.path.isfile(temp_path)==False:
                step=self.list_pcap.index(folder)
                if step+1 == len(self.list_pcap):
                    #self.last=os.listdir(self.list_pcap[self.list_pcap.index(folder)+1])[0]
                    break
                folder=self.list_pcap[step+1]
                name=os.path.split(folder)[1]+'_1.bin'
                nbr='1'
                nbr = str(int(nbr)+1)
                temp_path=os.path.join(folder,name)
            list_file_temp.append(temp_path)
        X, y = self.__data_generation(list_file_temp)
        self.last=list_file_temp[-1]
        print(list_file_temp[-1])
        return self.__transform_to_pytorch(X), torch.from_numpy(y)
    
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
    
    def __transform_to_pytorch(self, X):
        X = torch.from_numpy(X/255).float()
        X=X.unsqueeze(1)
        X=X.to(self.device)
        return X
        
class DataGenerator5(torch.utils.data.Dataset):
    
    def __init__(self, path, labels, N, split, t, shape, d, w):
        self.path=path
        self.labels=labels
        self.N=N
        self.split=split
        self.type=t
        self.shape = shape
        self.d = d
        self.w = w
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __len__(self):
        if self.type=='train':
            return self.N
        if self.type=='test':
            return self.N-int(self.N*self.split)
         
    def __getitem__(self, index):
        if self.type=='test':
            index+=int(self.N*self.split)
        ratio=int(np.prod(self.shape)/(2*self.d**2))
        ld=[os.path.join(self.path,d) for d in os.listdir(self.path)]
        for l in self.labels:
            find=False
            for day in ld:
                name=os.path.join(day,l)+'_'+str(index+1)+'.bin'
                if os.path.isfile(name) : 
                    sample=open(name, 'rb').read()
                    label=[1 if cl in name else 0 for cl in self.labels]
                    find = True
                    break
                else:
                    sample = None
                    label = None
            if find:
                break
        li=sample.hex()
        li=re.findall('..',li)
        li=li[:ratio] #We pick the n/d² first ones with n the size of the matrix
        li=li+['00']*(ratio-len(li)) #padding
        img=convert_sequ_to_image(li, self.d, self.w) #create the image from the sequences
        return self.__transform_to_pytorch(img), torch.from_numpy(np.asarray(label))
    
    def __transform_to_pytorch(self, X):
        X = torch.from_numpy(X/255).float()
        X=X.unsqueeze(0)
        X=X.to(self.device)
        return X
    
class DataGenerator6(torch.utils.data.Dataset):
    
    def __init__(self, path, labels, N, split, t, shape, d, w):
        self.path=path
        self.labels=labels
        self.split=int(N*split)
        self.type=t
        self.shape = shape
        self.d = d
        self.w = w
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if t=='train':
            self.N = self.split
        if t=='test':
            self.N = N - self.split
        self.list_files=[]
        for i in range(self.N):
                self.list_files.append(self.getdata(i))
                print(len(self.list_files))
    
    def __len__(self):
        return self.N
         
    def getdata(self, index):
        if self.type=='test':
            index+=self.split
        ratio=int(np.prod(self.shape)/(2*self.d**2))
        ld=[os.path.join(self.path,d) for d in os.listdir(self.path)]
        for l in self.labels:
            find=False
            for day in ld:
                name=os.path.join(day,l)+'_'+str(index+1)+'.bin'
                if os.path.isfile(name) : 
                    sample=open(name, 'rb').read()
                    label=[1 if cl in name else 0 for cl in self.labels]
                    find = True
                    break
                else:
                    sample = None
                    label = None
            if find:
                break
        li=sample.hex()
        li=re.findall('..',li)
        li=li[:ratio] #We pick the n/d² first ones with n the size of the matrix
        li=li+['00']*(ratio-len(li)) #padding
        img=convert_sequ_to_image(li, self.d, self.w) #create the image from the sequences
        return img, np.asarray(np.argmax(np.asarray(label),axis=0))
    
    def __getitem__(self, index):
        img, label = self.list_files[index]
        return self.transform_to_pytorch(img), torch.from_numpy(label)
    
    def transform_to_pytorch(self, X):
        X = torch.from_numpy(X/255).float()
        X=X.unsqueeze(0)
        X=X.to(self.device)
        return X
        
class DataGenerator_UDP(torch.utils.data.Dataset):
    
    def __init__(self, path, labels, N, shape, d, w):
        self.path=path
        self.labels=labels
        self.shape = shape
        self.list_name=glob.glob(path+'/*/*.bin')[:N]
        self.d = d
        self.w = w
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N=N

    def __len__(self):
        return len(self.list_name)
         
    def __getitem__(self, index):
        name = self.list_name[index]
        ratio=int(np.prod(self.shape)/(2*self.d**2))
        sample=open(name, 'rb').read()
        label=[1 if cl in name else 0 for cl in self.labels]
        li=sample.hex()
        li=re.findall('..',li)
        li=li[:ratio]
        li=li+['00']*(ratio-len(li))
        img=convert_sequ_to_image(li, self.d, self.w)
        label=np.asarray(label)
        return self.transform_to_pytorch(img), torch.from_numpy(label)
    
    def transform_to_pytorch(self, X):
        X = torch.from_numpy(X/255).float()
        X=X.unsqueeze(0)
        X=X.to(self.device)
        return X


