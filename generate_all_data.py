#!/usr/env/python3

import db

fileName = "all_data.csv"

_data = "No,TestSize,Precision,Recall,Accuracy,F1,Model\n"

handler = open(fileName, "w")
counter = 0
for data in db.fetchTable("all_data"):
    counter += 1
    _data += "{},{},{},{},{},{},{}\n".format(counter, data['tsize'], data['precision'], data['recall'], data['accuracy'], data['f1'], data['model'])

handler.write(_data)
handler.close()
