#!/usr/bin/python

"""
Visualizing data
"""

import db
import logger as lg
import matplotlib.pyplot as plt
import numpy as np
import func as fx

protocols = ['icmp', 'udp', 'tcp']
data = {'icmp': [[], [], [], []], 'udp': [[], [], [], []], 'tcp': [[], [], [], []]} # normal -> rx, tx  :: malicious -> rx, tx


sql = "select "
counter = 0
for x in range(1, 5):
    if counter == 0:
        sql += " max(p{}_rx_packets), max(p{}_tx_packets)".format(x, x)
    else:
        sql += " , max(p{}_rx_packets), max(p{}_tx_packets)".format(x, x)
    counter += 1
sql += " from data "

for proto in protocols:
    for scenario in range(0, 2):
        query = sql + " where protocol='{}' and label={}".format(proto, scenario)
        for res in db.query(query):
            res = [float(i) for i in res]

            if scenario == 0:
                # normal
                
                data[proto][0].append(res[0])
                data[proto][1].append(res[1])
                data[proto][0].append(res[2])
                data[proto][1].append(res[3])
                data[proto][0].append(res[4])
                data[proto][1].append(res[5])
                data[proto][0].append(res[6])
                data[proto][1].append(res[7])
            else:
                # malicious
                data[proto][2].append(res[0])
                data[proto][3].append(res[1])
                data[proto][2].append(res[2])
                data[proto][3].append(res[3])
                data[proto][2].append(res[4])
                data[proto][3].append(res[5])
                data[proto][2].append(res[6])
                data[proto][3].append(res[7])

# Generating graphs
for proto in protocols:
    plt.figure()
    plt.plot(fx.generatePoints(len(data[proto][0])), data[proto][0], '-bo')
    plt.plot(fx.generatePoints(len(data[proto][1])), data[proto][1], '-ro')
    print("Threshold: {} => {}, {}".format(proto, np.mean(data[proto][0]), np.mean(data[proto][1])))
    plt.title(proto + ' - Normal Rx Tx')
    
    plt.figure()
    plt.plot(fx.generatePoints(len(data[proto][2])), data[proto][2], '-bo')
    plt.plot(fx.generatePoints(len(data[proto][3])), data[proto][3], '-ro')
    print("Threshold: {} => {}, {}".format(proto, np.mean(data[proto][2]), np.mean(data[proto][3])))
    plt.title(proto + ' - Malicious Rx Tx')


#plt.show()
