"""
Detection
"""
import pprint
import logger as lg
import db
import numpy as np
import time 
import defense as df
import func as fx

def process(pkt, data, model, protocol, t_start):
    p_data = data
    model_name = 'knn'
    model = fx.loadLinearModel(model_name)
    model_output = int(model.predict(p_data)[0])
    #model_output = int(model.predict(np.reshape(data, (data.shape[0], 1, data.shape[1])))[0][0])
    d_time = time.time() - t_start
    if model_output == 1:
        db.addDD('defense', tuple([protocol, time.time() - t_start, model_name])) # i swapped db table name
        df.filter(pkt, protocol, t_start, model_name)
