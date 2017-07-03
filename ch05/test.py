# -*- coding: utf-8 -*-

import logRegres

dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.gradeAscent(dataArr, labelMat)
logRegres.plotBestFit(weights)
