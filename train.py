"""
Train Model
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import func as fx
import rnn_models as rmd
import cnn_models as cmd
import sys
import pprint
import logger as lg
import math
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pickle
import os
import db
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import *

def runScenario(data_file, test_size):
    # data source
    data_file = '{}.csv'.format(data_file)
    #data_file = 'data/ICMP_P2_Rx_Packet.csv'
    df = pd.read_csv(data_file, delimiter=',')
    #print(df.head())
    features_columns = []
    for x in range(len(fx.header) - 1):
        features_columns.append(x)
    features = df.iloc[:, features_columns].values
    target = df.iloc[:,[len(fx.header) - 1]].values
    X_train_default, X_test_default, y_train_default, y_test_default = train_test_split(features, target, random_state=0, test_size=test_size)
    X_train = X_train_default
    X_test = X_test_default
    y_train = y_train_default
    y_test = y_test_default
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # testing knn classifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lg.success('KNN: {:.4f}'.format(model.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
#fx.plot_cm(cm, title='KNN Confusion Matrix')
    fx.saveLinearModel('knn', model)
    db.addAllData(data_file, test_size, y_test, y_pred, 'knn')

    # testing logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lg.success('Logistic Regression: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Logistic Regression Confusion Matrix')
    fx.saveLinearModel('lr', model)
    db.addAllData(data_file, test_size, y_test, y_pred, 'lr')

    # testing linear svc
    model = LinearSVC()
    model.fit(X_train, y_train)
    lg.success('LinearSVC: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='LinearSVC Confusion Matrix')
    fx.saveLinearModel('lsvc', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'lsvc')

# testing svc
    model = SVC()
    model.fit(X_train, y_train)
    lg.success('SVC: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='SVC Confusion Matrix')
    fx.saveLinearModel('svc', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'svc')

# testing decision tree
    model = DecisionTreeClassifier(random_state=0, max_depth=7)
    model.fit(X_train, y_train)
    lg.success('Decision Tree: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Decision Tree Confusion Matrix')
    fx.saveLinearModel('dt', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'dt')

# testing random forest
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    lg.success('Random Forest: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Random Forest Confusion Matrix')
    fx.saveLinearModel('rf', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'rf')

# testing gradientboosting classifier
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    lg.success('Gradient Boosting: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Gradient Boosting Confusion Matrix')
    fx.saveLinearModel('gb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'gb')

# testing naive bayesian classifiers
    model = GaussianNB()
    model.fit(X_train, y_train)
    lg.success('Gaussian NB: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Gaussian NB Confusion Matrix')
    fx.saveLinearModel('gnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'gnb')

    model = BernoulliNB()
    model.fit(X_train, y_train)
    lg.success('Bernoulli NB: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Bernoulli NB Confusion Matrix')
    fx.saveLinearModel('bnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'bnb')

    model = MultinomialNB()
    model.fit(X_train, y_train)
    lg.success('Multinomial NB: {:.4f}'.format(model.score(X_test, y_test)))
#fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Multinomial NB Confusion Matrix')
    fx.saveLinearModel('mnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'mnb')

# neural network
    input_data_shape = X_train.shape[1]
    num_features = y_train.shape[1]
    batch_size=64
    epochs = 100
    X_train = np.reshape(X_train_default, (X_train_default.shape[0], 1, X_train_default.shape[1]))
    X_test = np.reshape(X_test_default, (X_test_default.shape[0], 1, X_test_default.shape[1]))

    lg.warning('\n\nNeural Network')
    lg.success("Num features: {}".format(num_features))

    model = rmd.lstm(input_data_shape, num_features)
    try:
        plot_model(model, to_file='./rnn_lstm.eps')
    except:
        pass
    plot_model(model, to_file='./rnn_lstm.png')
#model.summary(print_fn=fx.rnnprint)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    res = model.evaluate(X_test, y_test)
    lg.success('LSTM Accuracy: {:.4f}'.format(res[3]))
    y_pred = np.around(model.predict(X_test))
    y_pred = np.array([[int(i)] for i in y_pred])
#print(y_test, y_pred)
#fx.plot_cm(confusion_matrix(y_test, y_pred), title='LSTM Confusion Matrix')
#print('Test Loss: {:2f}'.format(res[2]))
#sys.exit()
# save model
    model.save('lstm_model.h5')
    lg.success('[+] Model saved')
    db.addAllData(data_file, test_size, y_test, y_pred, 'lstm')

# convolutional neural network
    lg.warning('\n\nCNN')
    X_train = np.reshape(X_train_default, (X_train_default.shape[0], X_train_default.shape[1], 1, 1))
    y_train = np.reshape(y_train_default, (y_train_default.shape[0], y_train_default.shape[1], 1, 1))

    X_test = np.reshape(X_test_default, (X_test_default.shape[0], X_test_default.shape[1], 1, 1))
    y_test = np.reshape(y_test_default, (y_test_default.shape[0], y_test_default.shape[1], 1, 1))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = cmd.default(X_train.shape[1:])
    try:
        plot_model(model, to_file='./cnn.eps')
    except:
        pass
    plot_model(model, to_file='./cnn.png')
#model.summary(print_fn=fx.cnnprint)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    res = model.evaluate(X_test, y_test)
    lg.success('CNN Accuracy: {:.4f}'.format(res[1]))
    y_pred = np.around(model.predict(X_test))
    y_pred = np.array([[int(i)] for i in y_pred])
#print(y_test, y_pred)
#fx.plot_cm(confusion_matrix(y_test_default, y_pred), title='CNN Confusion Matrix')
    model.save('cnn.h5')
    lg.success('[+] Model saved')
    db.addAllData(data_file, test_size, y_test_default, y_pred, 'cnn')


# running a set of scenarios
data_source = ['data']
splits = [0.2, 0.3, 0.4]
iterations = 20
for _ in range(iterations):
    for data in data_source:
        for split in splits:
            runScenario(data, split)
            sys.exit()
