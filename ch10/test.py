# -*- coding: utf-8 -*-

import kMeans
import numpy as np

def test1():
    dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
    print kMeans.randCent(dataMat, 2)
    print kMeans.distEclud(dataMat[0], dataMat[1])

test1()
