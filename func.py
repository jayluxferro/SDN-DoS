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
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from math import log10 as log
host_ip = '127.0.0.1'
host_port = 5000
results_path='results/'
host_iface = 'nat0-eth0'
api = 'http://localhost:8080'
ip_prefix = '10.0.0.'
switch_prefix = '00:00:00:00:00:00:00:0'
protocol_list = ['icmp', 'tcp', 'udp']
header=['Source IP', 'Destination IP', 'Protocol', 'Switch', 'P1', 'P2', 'P3', 'P4', 'P1 Rx Packet', 'P1 Tx Packet', 'P1 Rx Bytes', 'P1 Tx Bytes', 'P2 Rx Packet', 'P2 Tx Packet', 'P2 Rx Bytes', 'P2 Tx Bytes', 'P3 Rx Packet', 'P3 Tx Packet', 'P3 Rx Bytes', 'P3 Tx Bytes', 'P4 Rx Packet', 'P4 Tx Packet', 'P4 Rx Bytes', 'P4 Tx Bytes', 'Label']

def generatePoints(length):
    return np.linspace(1, length, length)

def saveLinearModel(prefix, model):
    file_name = results_path + prefix + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        lg.success('[+] Model save: ==> {}\n'.format(prefix))

def loadLinearModel(prefix):
    model = results_path + prefix + '.pkl'
    with open(model, 'rb') as file:
        return pickle.load(file)
    return None

def rnnprint(s):
    with open(results_path + 'rnn_modelsummary.txt','w+') as f:
        print(s, f)

def cnnprint(s):
    with open(results_path + 'cnn_modelsummary.txt','w+') as f:
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


def plotSinglePS(classifier, X_test, y_test, test_size):
    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    #disp.ax_.set_title('Precision-Recall curve: AP={0:0.2f} T={1:0.2f}'.format(disp.average_precision, test_size))
    #plt.show()
    plt.savefig(results_path + '{}_T_{}.eps'.format(disp.estimator_name, test_size))

def plotNPS(model, y_test, y_pred, test_size):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    #print(precision, recall, thresholds)
    plt.figure()
    plt.step(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    avgPrecision = np.average(precision)
    legend = 'Model={0} AP={1:0.2f} T={2:0.2f}'.format(model, np.average(precision), test_size)
    plt.legend([legend])
    #plt.title('Precision-Recall curve: Model={0} AP={1:0.2f}'.format(model, np.average(precision)))
    #plt.show()
    plt.savefig(results_path + '{}_T_{}.eps'.format(model, test_size))
    plt.savefig(results_path + '{}_T_{}.png'.format(model, test_size), dpi=1200)

    # plot the log graph
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision (log10)')
    plt.legend([legend])
    plt.step(recall, [log(i) for i in precision])
    plt.savefig(results_path + '{}_T_{}_log.eps'.format(model, test_size))
    plt.savefig(results_path + '{}_T_{}_log.png'.format(model, test_size), dpi=1200)
    print(precision, recall)
    return precision, recall, legend


def plotSummary(precisionList, recallList, legends, test_size):
    plt.figure()
    counter = 0
    for p in precisionList:
        plt.step(recallList[counter], precisionList[counter], label=legends[counter])
        counter += 1
    plt.legend(legends)
    plt.xlabel('Recall')
    plt.ylabel('Precision (log10)')
    plt.savefig(results_path + 'pr_summary_{}.eps'.format(test_size))
    plt.savefig(results_path + 'pr_summary_{}.png'.format(test_size), dpi=1200)
    
    # log graph
    plt.figure()
    counter = 0
    for p in precisionList:
        plt.step(recallList[counter], [log(i) for i in precisionList[counter]], label=legends[counter])
        counter += 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(legends)
    plt.savefig(results_path + 'pr_summary_{}_log.eps'.format(test_size))
    plt.savefig(results_path + 'pr_summary_{}_log.png'.format(test_size), dpi=1200)


def plotAllData(allData, index, modelLegend):
    plt.figure()
    counter = 0
    for d in allData:
        # 0.2
        node = d[index]
        plt.plot(node[0], node[1], '*', label=modelLegend[counter])
        plt.xlabel('Recall (%)')
        plt.ylabel('Precision (%)')
        plt.legend()
        counter += 1
    plt.show()

