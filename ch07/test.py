# -*- coding: utf-8 -*-

import adaboost
import numpy as np

datMat, classLabels = adaboost.loadSimpData()
D = np.mat(np.ones((5,1)) / 5)
bestStump, minError, bestClassEst = adaboost.buildStump(datMat, classLabels, D)
print(bestStump, minError, bestClassEst)
