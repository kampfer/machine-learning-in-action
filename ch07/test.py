# -*- coding: utf-8 -*-

import adaboost
import numpy as np

dataArr, classLabels = adaboost.loadSimpData()
classifierArr = adaboost.adaBoostTrainDS(dataArr, classLabels, 30)
print(adaboost.adaClassify([0,0], classifierArr))
