#!/usr/bin/python
"""
ICMP Packet analyzer
Date: 28 Jan 2019
"""
from scapy.all import *
import pprint
import logger as d
import db
import func

def process(pkt, label, t_start):
    ip = pkt.getlayer(IP)
    ether = pkt.getlayer(Ether)
    d.default(pprint.pformat(pkt))
    #func.addData(ip.src, ip.dst, "icmp", label)
    func.getDetectionData(pkt, ip.src, ip.dst, "icmp", label, t_start)
