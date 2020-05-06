#!/usr/bin/python
"""
Functions
"""

import subprocess
import db
import requests
from pandas import DataFrame as DF
import cm_pretty as cm_p
import numpy as np
import detection as dtn
import logger as lg
import pickle

host_ip = '127.0.0.1'
host_port = 5000
host_iface = 'nat0-eth0'
api = 'http://localhost:8080'
ip_prefix = '10.0.0.'
switch_prefix = '00:00:00:00:00:00:00:0'
protocol_list = ['icmp', 'tcp', 'udp']
header=['Source IP', 'Destination IP', 'Protocol', 'Switch', 'P1', 'P2', 'P3', 'P4', 'P1 Rx Packet', 'P1 Tx Packet', 'P1 Rx Bytes', 'P1 Tx Bytes', 'P2 Rx Packet', 'P2 Tx Packet', 'P2 Rx Bytes', 'P2 Tx Bytes', 'P3 Rx Packet', 'P3 Tx Packet', 'P3 Rx Bytes', 'P3 Tx Bytes', 'P4 Rx Packet', 'P4 Tx Packet', 'P4 Rx Bytes', 'P4 Tx Bytes', 'Label']

def generatePoints(length):
    return np.linspace(1, length, length)

def saveLinearModel(prefix, model):
    file_name = prefix + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        lg.success('[+] Model save: ==> {}\n'.format(prefix))

def loadLinearModel(prefix):
    model = prefix + '.pkl'
    with open(model, 'rb') as file:
        return pickle.load(file)
    return None

def rnnprint(s):
    with open('rnn_modelsummary.txt','w+') as f:
        print(s, f)

def cnnprint(s):
    with open('cnn_modelsummary.txt','w+') as f:
        print(s, f)

def plot_cm(cm, title='Confusion Matrix'):
    cmap = 'PuRd'
    cm = np.array(cm)
    cm_p.pretty_plot_confusion_matrix(DF(cm), cmap=cmap, title=title)

def formatProtocol(protocol):
    return protocol_list.index(protocol)

def formatIP(ip):
    return int(ip.split(ip_prefix)[-1])

def formatSwitch(switch):
    return int(switch.split(switch_prefix)[-1])

def inSubnet(ip):
    subnet = db.getSubnet()['ip'].encode('utf-8')
    # classful
    subnet = subnet.split('/')
    c = subnet[-1]
    r = int(int(c)/8)
    ip = ip.split('.')
    subnet = subnet[0].split('.')
    if subnet[:r] == ip[:r]:
        return True
    return False

def setSubnet(iface):
    sw = subprocess.check_output(['ip','addr','show',iface], shell=False)
    gw = sw.splitlines()[2].split(' ')[5]
    mac = sw.splitlines()[1].split(' ')[5]
    db.setSubnet(gw, mac)


def getData(url):
    res = requests.get(api + url)
    if res.status_code == 200:
        return res.json()
    return None

def getSwitchData():
    return getData('/wm/core/switch/all/port/json')

def addData(source_ip, destination_ip, protocol, label):
    switchData = getSwitchData()
    if switchData is not None:
        # there's data okay oo
        for switch in switchData:
            data = []
            data.append(source_ip) # source ip
            data.append(destination_ip) # destination ip
            data.append(protocol) # protocol
            data.append(switch) # switch_mac
            port_info = switchData[switch]['port_reply'][0]['port']
            p1_data = port_info[0]
            p2_data = port_info[1]
            p3_data = port_info[2]
            p4_data = port_info[3]

            # p1, p2, p3, p4
            data.append(0)
            data.append(1)
            data.append(2)
            data.append(3)

            # p1 data
            data.append(p1_data['receive_packets'])
            data.append(p1_data['transmit_packets'])
            data.append(p1_data['receive_bytes'])
            data.append(p1_data['transmit_bytes'])
            
            # p2 data
            data.append(p2_data['receive_packets'])
            data.append(p2_data['transmit_packets'])
            data.append(p2_data['receive_bytes'])
            data.append(p2_data['transmit_bytes'])

            # p3 data
            data.append(p3_data['receive_packets'])
            data.append(p3_data['transmit_packets'])
            data.append(p3_data['receive_bytes'])
            data.append(p3_data['transmit_bytes'])


            # p4 data
            data.append(p4_data['receive_packets'])
            data.append(p4_data['transmit_packets'])
            data.append(p4_data['receive_bytes'])
            data.append(p4_data['transmit_bytes'])

            # label
            data.append(label)

            # add data to db
            db.addData(data)

def getDetectionData(pkt, source_ip, destination_ip, protocol, label, t_start):
    switchData = getSwitchData()
    if switchData is not None:
        # there's data okay oo
        for switch in switchData:
            data = []
            data.append(int(source_ip[-1])) # source ip
            data.append(int(destination_ip[-1])) # destination ip
            data.append(formatProtocol(protocol)) # protocol
            data.append(int(switch[-1])) # switch_mac
            port_info = switchData[switch]['port_reply'][0]['port']
            p1_data = port_info[0]
            p2_data = port_info[1]
            p3_data = port_info[2]
            p4_data = port_info[3]

            # p1, p2, p3, p4
            data.append(0)
            data.append(1)
            data.append(2)
            data.append(3)

            # p1 data
            data.append(int(p1_data['receive_packets']))
            data.append(int(p1_data['transmit_packets']))
            data.append(int(p1_data['receive_bytes']))
            data.append(int(p1_data['transmit_bytes']))
            
            # p2 data
            data.append(int(p2_data['receive_packets']))
            data.append(int(p2_data['transmit_packets']))
            data.append(int(p2_data['receive_bytes']))
            data.append(int(p2_data['transmit_bytes']))

            # p3 data
            data.append(int(p3_data['receive_packets']))
            data.append(int(p3_data['transmit_packets']))
            data.append(int(p3_data['receive_bytes']))
            data.append(int(p3_data['transmit_bytes']))


            # p4 data
            data.append(int(p4_data['receive_packets']))
            data.append(int(p4_data['transmit_packets']))
            data.append(int(p4_data['receive_bytes']))
            data.append(int(p4_data['transmit_bytes']))

            # forward data to detection engine
            dtn.process(pkt, np.array([data]), label, protocol, t_start)
