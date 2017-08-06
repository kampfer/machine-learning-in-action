# -*- coding: utf-8 -*-

import svmMLiA
import numpy as np

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
# b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
# print(b)
# print(alphas[alphas > 0])
# for i in range(100):
#    if alphas[i] > 0:
#        print(dataArr[i], labelArr[i])
ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
dataMat = np.mat(dataArr)
m = np.shape(dataMat)[0]
j = svmMLiA.selectJrand(0, m)
print(j, dataMat[j] * np.mat(ws) + b, labelArr[j])
