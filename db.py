#!/usr/bin/python

import logger as log
import sqlite3
import func 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def init():
    conn = sqlite3.connect('test.db')
    conn.row_factory = sqlite3.Row
    return conn

def setSubnet(ip, mac):
    db = init()
    cursor = db.cursor()
    cursor.execute("update subnet set ip=?, mac=?", (ip,mac))
    db.commit()
    log.success('Updated subnet')

def query(sql):
    db = init()
    cursor = db.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

def getSubnet():
    db = init()
    cursor = db.cursor()
    cursor.execute("select * from subnet limit 1")
    return cursor.fetchone()

def addData(data):
    db = init()
    cursor = db.cursor()
    cursor.execute("insert into data(source_ip, destination_ip, protocol, switch_mac, p1, p2, p3, p4, p1_rx_packets, p1_tx_packets, p1_rx_bytes, p1_tx_bytes, p2_rx_packets, p2_tx_packets, p2_rx_bytes, p2_tx_bytes, p3_rx_packets, p3_tx_packets, p3_rx_bytes, p3_tx_bytes, p4_rx_packets, p4_tx_packets, p4_rx_bytes, p4_tx_bytes, label) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(data))
    db.commit()
    log.success('[+] Added new data')

def fetchTable(tableName):
    db = init()
    cursor = db.cursor()
    cursor.execute('select * from {}'.format(tableName))
    return cursor.fetchall()

def addDD(table, data):
    db = init()
    cursor = db.cursor()
    cursor.execute("insert into "+ table + "(protocol, time, model) values(?, ?, ?)", tuple(data))
    db.commit()
    log.success('[+] {} data added'.format(table))

def addAllData(data, tsize, y_test, y_pred, model):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred)
    db = init()
    cursor = db.cursor()
    cursor.execute("insert into all_data(data, tsize, precision, recall, accuracy, model) values(?, ?, ?, ?, ?, ?)", (data, tsize, precision, recall, accuracy, model))
    db.commit()
    log.success('[+] {} <=> {} <=> {}'.format(data, tsize, model))
