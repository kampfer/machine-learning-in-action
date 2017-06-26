# -*- coding: utf-8 -*-

import math

# 计算信息熵
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

# 从dataSet中筛选出指定列axis的值等于value的行
# 筛选出的行将删去列axis
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for vec in dataSet:
        if vec[axis] == value:
            reducedVec = vec[:axis]
            reducedVec.extend(vec[axis + 1:])
            retDataSet.append(reducedVec)
    return retDataSet

# 找出信息增益最大的划分方式
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
            # 每个特征的不同特征值的信息熵的加权和，权重是特征值的频率
            # （熵越大，代表信息越多，即不同的特征值越多？可以这么理解？）
            newEntropy += prob * calShannonEnt(subDataSet)
        # 信息增益是熵的减少或者是数据无序度的减少
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[key] = 0
        classCount[key] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    del(labels[bestFeat])
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels[:])
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
