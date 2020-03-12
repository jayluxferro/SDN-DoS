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

# debugging
import pprint

# inits
sio = socketio.Client()
server='http://{}:{}'.format(func.host_ip, func.host_port)

def usage():
    print('Usage: python {} <scenario>'.format(sys.argv[0]))
    sys.exit(1)

# packet handler
def packetHandler(pkt):
    global mac
    global scenario
    #d.warning(pprint.pformat(pkt))

    if pkt.haslayer(IP):
        ip = pkt.getlayer(IP)
        ether = pkt.getlayer(Ether)
        
        if ip.dst != None and func.inSubnet(ip.dst) and ether.src != mac: # removing data from AP
            if pkt.haslayer(ICMP):
                icmp.process(pkt, scenario)

            if pkt.haslayer(UDP):
                udp.process(pkt, scenario)

            if pkt.haslayer(TCP):
                tcp.process(pkt, scenario)

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
