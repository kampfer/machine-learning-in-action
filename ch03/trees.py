# -*- coding: utf-8 -*-

import math

# 信息熵越大，信息量越大
def calShannonEnt(dataSet):
    labels = {}
    for vec in dataSet:
        currentLabel = vec[-1]
        if currentLabel not in labels.keys():
            labels[currentLabel] = 0
        labels[currentLabel] += 1
    shannonEnt = 0.0
    lenDataSet = len(dataSet)
    for key in labels:
        # 两个整数相除，最后的结果会被转换成整数
        # 所以当结果是小数时，会被转成0！！！
        # 这里先转成浮点数，是为了避免这点！！
        prob = float(labels[key]) / lenDataSet
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flipers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = {}
    for vec in dataSet:
        if vec[axis] == value:
            reducedVec = vec[:axis]
            reducedVec.extend(vec[axis + 1:])
            retDataSet.append(reducedVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
