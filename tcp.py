#!/usr/bin/python
"""
TCP packet analyzer
Date: 28th Jan 2019
"""
from scapy.all import *
import pprint
import logger as d
import db
import func

def process(pkt, label, t_start):
    ip = pkt.getlayer(IP)
    ether = pkt.getlayer(Ether)
    d.warning(pprint.pformat(pkt))
    stack = pkt.getlayer(TCP)
    #func.addData(ip.src, ip.dst, "tcp", label)

    func.getDetectionData(pkt, ip.src, ip.dst, "tcp", label, t_start)
