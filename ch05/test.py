# -*- coding: utf-8 -*-

import logRegres
import numpy as np

def colicTest():
    return 0

def multiTest():
    return 0

dataArr, labelMat = logRegres.loadDataSet()

# weights = logRegres.gradeAscent(dataArr, labelMat)
# logRegres.plotBestFit(weights.getA())

# weights = logRegres.stocGradAscent0(np.array(dataArr), labelMat)
weights = logRegres.stocGradAscent1(np.array(dataArr), labelMat)
logRegres.plotBestFit(weights)
