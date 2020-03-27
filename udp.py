#!/usr/bin/python
"""
UDP Packet Analyzer
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
    d.error(pprint.pformat(pkt))
    stack = pkt.getlayer(UDP)
    #func.addData(ip.src, ip.dst, "udp", label)
    func.getDetectionData(pkt, ip.src, ip.dst, "udp", label, t_start)
