#!/usr/bin/python
"""
Functions
"""

import subprocess
import db
import requests

host_ip = '127.0.0.1'
host_port = 5000
host_iface = 'nat0-eth0'
api = 'http://localhost:8080'

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
