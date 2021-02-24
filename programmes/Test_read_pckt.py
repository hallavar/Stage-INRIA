# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:36:24 2021

@author: Utilisateur
"""

import glob

from utils import create_checkpoint
from data_extraction import batch_of_raw_pkt

create_checkpoint()

file_path='../dataset/cic-ids-2018/Original Network Traffic and Log data'
file_list=glob.glob(file_path+'/*/pcap/*')
file =file_list[3]


def verify_pcap_organization(pcap):
    a=pcap.read_packet().time
    b=pcap.read_packet().time
    while b-a>=0:
        a=b
        b=pcap.read_packet().time
        print(b-a)
        
batch=batch_of_raw_pkt(file, 64, 'hex')
print(batch)