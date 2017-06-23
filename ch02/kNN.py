import numpy as np
import operator
import pandas as pd

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def autoNorm(data):
    maxVals = data.max(axis=0)
    minVals = data.min(axis=0)
    ranges = maxVals - minVals
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData, ranges, minVals

def classifyPerson():
    resultList = ['not at all',  'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frquent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream comsumed per year?"))
    data = pd.read_table('datingTestSet2.txt', names=['a', 'b', 'c', 'd'])
    normData, ranges, minVals = autoNorm(data.iloc[:,:-1])
    datingLabels = data.d
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normData, datingLabels, 3)
    print "you will probably like this person: ", resultList[classifierResult - 1]

