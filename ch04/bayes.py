# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 提取出数据中所有的单词，并去重
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet |= set(document)
    return list(vocabSet)

# 判断指定列表中的每个单词是否在数据集中出现过
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numVocab = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 为了保证概率不为0，所以这里不能用0初始化
    # p0Num = np.zeros(numVocab)
    # p1Num = np.zeros(numVocab)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = np.ones(numVocab)   # 所有单词在分类为0的文档中出现的次数
    p1Num = np.ones(numVocab)   # 所有单词在分类为1的文档中出现的次数
    p0Denom = 2.0   # 
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 为了防止概率太小导致下溢出所以这里取对数
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClassl):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClassl)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClassl)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    postList, classList = loadDataSet()
    myVocabList = createVocabList(postList)
    trainMat = []
    for post in postList:
        trainMat.append(setOfWords2Vec(myVocabList, post))
    p0V, p1V, pAb = trainNB0(trainMat, classList)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
