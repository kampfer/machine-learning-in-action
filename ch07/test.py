# -*- coding: utf-8 -*-

import adaboost
import numpy as np

# dataArr, classLabels = adaboost.loadSimpData()
# classifierArr = adaboost.adaBoostTrainDS(dataArr, classLabels, 30)
# print(adaboost.adaClassify([0,0], classifierArr))

dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArr, aggClassEst = adaboost.adaBoostTrainDS(dataArr, labelArr, 10)
adaboost.plotROC(aggClassEst.T, labelArr)
# testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
# prediction10 = adaboost.adaClassify(testArr, classifierArr)
# errArr = np.mat(np.ones((67,1)))
# print(errArr[prediction10 != np.mat(testLabelArr).T].sum())
