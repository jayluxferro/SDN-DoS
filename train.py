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

# data source
data_file = 'data.csv'
df = pd.read_csv(data_file, delimiter=',')
features_columns = []
for x in range(len(fx.header) - 1):
    features_columns.append(x)
features = df.iloc[:, features_columns].values
target = df.iloc[:,[len(fx.header) - 1]].values

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# testing knn classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
lg.success('KNN: {:.2f}'.format(model.score(X_test, y_test)))
cm = confusion_matrix(y_test, y_pred)
fx.plot_cm(cm, title='KNN Confusion Matrix')
fx.saveLinearModel('knn', model)

# testing logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
lg.success('Logistic Regression: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Logistic Regression Confusion Matrix')
fx.saveLinearModel('lr', model)

# testing linear svc
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
lg.success('LinearSVC: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='LinearSVC Confusion Matrix')
fx.saveLinearModel('lsvc', model)

# testing svc
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
lg.success('SVC: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='SVC Confusion Matrix')
fx.saveLinearModel('svc', model)

# testing decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0, max_depth=7)
model.fit(X_train, y_train)
lg.success('Decision Tree: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Decision Tree Confusion Matrix')
fx.saveLinearModel('dt', model)

# testing random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
lg.success('Random Forest: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Random Forest Confusion Matrix')
fx.saveLinearModel('rf', model)

# testing gradientboosting classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
lg.success('Gradient Boosting: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Gradient Boosting Confusion Matrix')
fx.saveLinearModel('gb', model)

# testing naive bayesian classifiers
from sklearn.naive_bayes import *
model = GaussianNB()
model.fit(X_train, y_train)
lg.success('Gaussian NB: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Gaussian NB Confusion Matrix')
fx.saveLinearModel('gnb', model)

model = BernoulliNB()
model.fit(X_train, y_train)
lg.success('Bernoulli NB: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Bernoulli NB Confusion Matrix')
fx.saveLinearModel('bnb', model)

model = MultinomialNB()
model.fit(X_train, y_train)
lg.success('Multinomial NB: {:.2f}'.format(model.score(X_test, y_test)))
fx.plot_cm(confusion_matrix(y_test, model.predict(X_test)), title='Multinomial NB Confusion Matrix')
fx.saveLinearModel('mnb', model)

#sys.exit(1)
# neural network
batch_size=64
epochs = 100
num_features = y_train.shape[1]
input_data_shape = X_train.shape[1]
lg.warning('\n\nNeural Network')
lg.success("Num features: {}".format(num_features))

shap3 = 100
div = int(math.sqrt(shap3))

X_train_1 = np.resize(X_train, (X_train.shape[1] * 10 , X_train.shape[1] * 10))
X_test_1 = np.resize(X_test, (X_test.shape[1] * 10, X_test.shape[1] * 10 ))
print(X_train_1.shape, X_test_1.shape, div)


X_train_1 = np.reshape(X_train_1, (shap3, int(X_train_1.shape[1]/div), int(X_train_1.shape[1]/div), 1))
X_test_1 = np.reshape(X_test_1, (shap3, int(X_test_1.shape[1]/div), int(X_test_1.shape[1]/div), 1))
print(X_train_1.shape)
y_train_1 = np.resize(y_train, (y_train.shape[1] * div, y_train.shape[1] * div))
y_train_1 = np.reshape(y_train_1, (shap3,))

y_test_1 = np.resize(y_test, (y_test.shape[1] * div, y_test.shape[1] * div))
y_test_1 = np.reshape(y_test_1, (shap3,))

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = rmd.lstm(input_data_shape, num_features)
try:
    plot_model(model, to_file='./rnn_lstm.eps')
except:
    pass
plot_model(model, to_file='./rnn_lstm.png')
#model.summary(print_fn=fx.rnnprint)
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
res = model.evaluate(X_test, y_test)
lg.success('LSTM Accuracy: {:.2f}'.format(res[3]))
y_pred = np.around(model.predict(X_test))
y_pred = np.array([[int(i)] for i in y_pred])
#print(y_test, y_pred)
#fx.plot_cm(confusion_matrix(y_test, y_pred), title='LSTM Confusion Matrix')
#print('Test Loss: {:2f}'.format(res[2]))
#sys.exit()

# save model
model.save('lstm_model.h5')
lg.success('[+] Model saved')

# convolutional neural network
lg.warning('\n\nCNN')
X_train = X_train_1
X_test = X_test_1
y_train = y_train_1
y_test = y_test_1
print(y_train, len(y_train))

print(X_train.shape)
model = cmd.default(X_train.shape[1:])
try:
    plot_model(model, to_file='./cnn.eps')
except:
    pass
plot_model(model, to_file='./cnn.png')
#model.summary(print_fn=fx.cnnprint)

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
res = model.evaluate(X_test, y_test)
lg.success('CNN Accuracy: {:.2f}'.format(res[1]))
y_pred = np.around(model.predict(X_test))
y_pred = np.array([[int(i)] for i in y_pred])
#print(y_test, y_pred)
fx.plot_cm(confusion_matrix(y_test, y_pred), title='CNN Confusion Matrix')
model.save('lstm_model.h5')
lg.success('[+] Model saved')
