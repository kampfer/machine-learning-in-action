# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import regression

def test1():
    xArr, yArr = regression.loadDataSet('ex0.txt')
    ws = regression.standRegress(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    x = xMat.copy()
    x.sort(0)
    y = x * ws
    ax.plot(x[:,1], y)
    plt.show()

def test2():
    xArr, yArr = regression.loadDataSet('ex0.txt')
    yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

# test1()
test2()
