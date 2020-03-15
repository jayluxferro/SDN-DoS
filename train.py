"""
Train Model
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import func as fx
import rnn_models as rmd
import cnn_models as cmd
import sys
import pprint
import logger as lg
import math

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
"""
y_train = y_train.reshape(1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
sys.exit()
"""


# testing knn classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
lg.success('KNN: {:.2f}'.format(model.score(X_test, y_test)))

# testing logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
lg.success('Logistic Regression: {:.2f}'.format(model.score(X_test, y_test)))

# testing linear svc
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
lg.success('LinearSVC: {:.2f}'.format(model.score(X_test, y_test)))


# testing svc
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
lg.success('SVC: {:.2f}'.format(model.score(X_test, y_test)))


# testing decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0, max_depth=7)
model.fit(X_train, y_train)
lg.success('Decision Tree: {:.2f}'.format(model.score(X_test, y_test)))

# testing random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
lg.success('Random Forest: {:.2f}'.format(model.score(X_test, y_test)))

# testing gradientboosting classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
lg.success('Gradient Boosting: {:.2f}'.format(model.score(X_test, y_test)))


# testing naive bayesian classifiers
from sklearn.naive_bayes import *
model = GaussianNB()
model.fit(X_train, y_train)
lg.success('Gaussian NB: {:.2f}'.format(model.score(X_test, y_test)))

model = BernoulliNB()
model.fit(X_train, y_train)
lg.success('Bernoulli NB: {:.2f}'.format(model.score(X_test, y_test)))

model = MultinomialNB()
model.fit(X_train, y_train)
lg.success('Multinomial NB: {:.2f}'.format(model.score(X_test, y_test)))

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

X_train_1 = np.reshape(X_train_1, (shap3, X_train_1.shape[1]/div, X_train_1.shape[1]/div, 1))
X_test_1 = np.reshape(X_test_1, (shap3, X_test_1.shape[1]/div, X_test_1.shape[1]/div, 1))
print(X_train_1.shape)
y_train_1 = np.resize(y_train, (y_train.shape[1] * div, y_train.shape[1] * div))
y_train_1 = np.reshape(y_train_1, (shap3,))

y_test_1 = np.resize(y_test, (y_test.shape[1] * div, y_test.shape[1] * div))
y_test_1 = np.reshape(y_test_1, (shap3,))

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = rmd.lstm(input_data_shape, num_features)
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
res = model.evaluate(X_test, y_test)
lg.success('LSTM Accuracy: {:.2f}'.format(res[3]))
#print('Test Loss: {:2f}'.format(res[2]))

# convolutional neural network
lg.warning('\n\nCNN')
X_train = X_train_1
X_test = X_test_1
y_train = y_train_1
y_test = y_test_1
print(y_train, len(y_train))

print(X_train.shape)
model = cmd.default(X_train.shape[1:])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
res = model.evaluate(X_test, y_test)
lg.success('CNN Accuracy: {:.2f}'.format(res[1]))

