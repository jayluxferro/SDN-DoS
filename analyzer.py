#!/usr/bin/python
"""
Author: Jay Lux Ferro
Date:   9th February, 2020
Task:   Packet analyzer for DoS
"""

import sys
import logger as d
from scapy.all import *
import socketio
import func
import db
import time
import icmp, udp, tcp
import rnn_models as rmd

# debugging
import pprint

# inits
t_start = time.time()
sio = socketio.Client()
server='http://{}:{}'.format(func.host_ip, func.host_port)
num_features = 1
input_data_shape = 24
model = rmd.lstm(input_data_shape, num_features)
model.load_weights('./model.h5')

def usage():
    print('Usage: python {} <scenario>'.format(sys.argv[0]))
    sys.exit(1)

# packet handler
def packetHandler(pkt):
    global mac
    global scenario
    #d.warning(pprint.pformat(pkt))
    
    # passing model as second param
    scenario = model
    global t_start

    if pkt.haslayer(IP):
        ip = pkt.getlayer(IP)
        ether = pkt.getlayer(Ether)
        
        if ip.dst != None and func.inSubnet(ip.dst) and ether.src != mac: # removing data from AP
            if pkt.haslayer(ICMP):
                icmp.process(pkt, scenario, t_start)

            if pkt.haslayer(UDP):
                udp.process(pkt, scenario, t_start)

            if pkt.haslayer(TCP):
                tcp.process(pkt, scenario, t_start)

# entry
if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()

    # connect to rpc
    sio.connect(server)
    
    func.setSubnet(func.host_iface)
    global mac
    global scenario
    scenario = sys.argv[1]
    mac = db.getSubnet()['mac']

    # sniff for packets
    d.default('[+] Analyzing traffic on {}'.format(func.host_iface))
    while True:
        sniff(iface=func.host_iface, count=1, prn=packetHandler)
