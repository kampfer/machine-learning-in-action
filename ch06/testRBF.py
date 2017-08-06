# -*- coding: utf-8 -*-

import svmMLiA
import numpy as np

def testRbf(k1=1.3):
    dataArr, labelArr = svmMLiA.loadDataSet('testSetRBF.txt')
    b, alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))

    dataArr, labelArr = svmMLiA.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))

testRbf();
