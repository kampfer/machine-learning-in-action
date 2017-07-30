# -*- coding: utf-8 -*-

import svmMLiA

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

