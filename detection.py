"""
Detection
"""
import pprint
import logger as lg
import db
import numpy as np
import time 
import defense as df

def process(pkt, data, model, protocol, t_start):
    output = int(model.predict(np.reshape(data, (data.shape[0], 1, data.shape[1])))[0][0])
    d_time = time.time() - t_start
    if output == 1:
        db.addDD('detection', tuple([protocol, time.time() - t_start]))
        df.filter(pkt, protocol, t_start) 
