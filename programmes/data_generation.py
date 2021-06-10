    # -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:26:02 2021

@author: Utilisateur
"""
import glob
import os.path
import scapy.all
import pandas as pd
from data_extraction import convert_pkt_to_bytes_sequ

def create_dataset_from_pcap(path_source, path_dest):
    list_pcap=glob.glob(path_source+'/*/*/*')
    j=0
    for name in list_pcap:
        pcap=scapy.utils.PcapReader(name)
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
            if pkt[2].layers==2 and pkt.haslayer('UDP'):
                sequence=convert_pkt_to_bytes_sequ(pkt, suppr=None)
                sequence=bytes(pkt)
                name=filename+str(j)+'.bin'
                print(name)
                with open(name, 'wb') as file:
                    file.write(sequence)
                    
def create_flows_csv(path_source, path_dest):
    list_pcap=glob.glob(path_source+'/*/*/*')
    session=dict({})
    for name in list_pcap:
        pcap=scapy.utils.rdpcap(name)
        session.update(pcap.sessions())
    udp=[k for k in session.keys() if 'UDP' in k]
    udp = [session[k] for k in udp if 'IP' in session[k][0]]
    
    Duration=[]
    Bytes=[]
    Packets=[]
    Timestamp=[]
    initial_ID=[]
    TTL=[]
    sport=[]
    dport=[]
    
    for flow in udp:
            Duration.append(float(flow[len(flow)-1].time - flow[0].time))
            Packets.append(len(flow))
            Timestamp.append(float(flow[0].time))
            initial_ID.append(flow[0].id)
            TTL.append(flow[0].ttl)
            sport.append(flow[0].sport)
            dport.append(flow[0].dport)
            l=0
            for pkt in flow:
                l+=len(pkt[3])
            Bytes.append(l)

    dic=dict({'initial_ID':initial_ID, 'Timestamp':Timestamp, 'Duration':Duration, 'Bytes':Bytes, 'Packets':Packets, 'TTL':TTL, 'source_port':sport, 'dest_port':dport})
    csv=pd.DataFrame(dic)
    csv.to_csv(path_dest+'/test.csv')
               
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
                    
if __name__=='__main__':
    print('u')
    create_flows_csv('/srv/tempdd/aschoen/labelled','/srv/tempdd/aschoen')
