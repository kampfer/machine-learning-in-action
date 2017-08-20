# -*- coding: utf-8 -*-

import kMeans
import numpy as np

def test1():
    dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
    print kMeans.randCent(dataMat, 2)
    print kMeans.distEclud(dataMat[0], dataMat[1])

def test2():
    dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
    myCentroids, clusterAssing = kMeans.kMeans(dataMat, 4)
    print(clusterAssing)

def test3():
    dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
    kMeans.biKmeans(dataMat, 4)

def test4():
    dataMat = np.mat(kMeans.loadDataSet('testSet2.txt'))
    centList, myNewAssments = kMeans.biKmeans(dataMat, 3)
    print centList

test4()
