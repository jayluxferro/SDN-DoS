"""
All Data Analysis
"""

import db
import func as fx

models = ['knn', 'lr', 'lsvc', 'svc', 'dt', 'rf', 'gb', 'gnb', 'bnb', 'mnb', 'lstm', 'cnn']
modelLegend = ['KNN', 'LogisticRegression', 'LinearSVC', 'SVC', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'GaussianNB', 'BernoulliNB', 'MultinomialNB', 'LSTM', 'CNN']
splitRatios = [0.2, 0.3, 0.4]

allData = [[[[] for _ in range(2)] for _ in range(len(splitRatios))] for _ in range(len(models))] # [model] => [split] => [[recall], [precision]]

for d in db.fetchTable('all_data'):
    modelIndex = models.index(d['model'])
    splitIndex = splitRatios.index(d['tsize'])
    recall = d['recall']
    precision = d['precision']
    node = allData[modelIndex][splitIndex]
    node[0].append(recall)
    node[1].append(precision)


for x in splitRatios:
    fx.plotAllData(allData, splitRatios.index(x), modelLegend)
    #fx.plotAllDataPrecision(allData, splitRatios.index(x), models, modelLegend)
    #fx.plotAllDataRecall(allData, splitRatios.index(x), models, modelLegend)

