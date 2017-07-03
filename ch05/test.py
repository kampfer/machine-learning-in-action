# -*- coding: utf-8 -*-

import logRegres
import numpy as np

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logRegres.stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(logRegres.classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print 'the error rate of this test is: %f' % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests))

dataArr, labelMat = logRegres.loadDataSet()

# weights = logRegres.gradeAscent(dataArr, labelMat)
# logRegres.plotBestFit(weights.getA())

# weights = logRegres.stocGradAscent0(np.array(dataArr), labelMat)
# weights = logRegres.stocGradAscent1(np.array(dataArr), labelMat)
# logRegres.plotBestFit(weights)

multiTest()
