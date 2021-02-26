# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:36:24 2021

@author: Utilisateur
"""

import glob
from datetime import datetime
from utils import create_checkpoint
from data_extraction import DataGenerator

create_checkpoint()

file_path='E:/stageINRIA/dataset/cic-ids-2018/labelled'
file_list=glob.glob(file_path+'/*/*/pcap/*')
file =file_list[3]


def verify_pcap_organization(pcap):
    a=pcap.read_packet().time
    b=pcap.read_packet().time
    while b-a>=0:
        a=b
        b=pcap.read_packet().time
        print(b-a)

def get_pckt_time(pkt):
    time=datetime.fromtimestamp(float(pkt.time))
    return str(time)

img_size = (28, 28)

generator = DataGenerator(file_path, img_size, 2, 16, 34, 512, 2)