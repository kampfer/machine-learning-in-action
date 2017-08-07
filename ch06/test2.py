# -*- coding: utf-8 -*-

import svmMLiA
import numpy as np

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = svmMLiA.loadImages('digits/trainingDigits')
    b,alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % np.shape(sVs)[0]
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = svmMLiA.loadImages('digits/testDigits')
    errorCount = 0
    datMat=np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)

testDigits();
