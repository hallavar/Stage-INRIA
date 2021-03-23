# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:26:02 2021

@author: Utilisateur
"""
import glob
import os.path
from scapy.all import PcapReader
from data_extraction import convert_pkt_to_bytes_sequ

def create_dataset_from_pcap(path_source, path_dest):
    list_pcap=glob.glob(path_source+'/*/*/*')
    j=0
    for name in list_pcap:
        pcap=PcapReader(name)
        name=os.path.normpath(name)
        name=name.split(os.sep)[-3:]
        try:
            os.mkdir(os.path.join(path_dest,name[:-2]))
        except:
            pass
        filename=os.path.join(path_dest,name[-2],name[-3]+'_')
        while True:
            try:
                pkt=pcap.read_packet()
                j+=1
            except EOFError:
                break
            sequence=convert_pkt_to_bytes_sequ(pkt, suppr=None)
            sequence=bytes(pkt)
            name=filename+str(j)+'.bin'
            print(name)
            with open(name, 'wb') as file:
                file.write(sequence)
                
def create_empty_dirtree(srcdir, dstdir, onerror=None):
    srcdir = os.path.abspath(srcdir)
    srcdir_prefix = len(srcdir) + len(os.path.sep)
    for root, dirs, files in os.walk(srcdir, onerror=onerror):
        for dirname in dirs:
            dirpath = os.path.join(dstdir, root[srcdir_prefix:], dirname)
            try:
                os.mkdir(dirpath)
            except OSError as e:
                if onerror is not None:
                    onerror(e)
