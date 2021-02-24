# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:26:39 2021

@author: Utilisateur
"""
import re
import numpy as np
from scapy.all import PcapReader #Maybe you will have to use PcapNgReader instead
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

def convert_pkt_to_image(pkt,d,w):
    pkt=bytes(pkt).hex() #Converting the packet to hexadecimal
    li=re.findall('..',pkt) #Get a list of bytes
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

def batch_of_pkt_img(file, d, batch_size, w):
    pcap=PcapReader(file)
    batch=[]
    for i in range(0,batch_size):
        pkt=pcap.read_packet()
        img=convert_pkt_to_image(pkt,d,w)
        batch.append(img)
    return np.asarray(batch, dtype=object)