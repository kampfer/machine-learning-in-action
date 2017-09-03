# -*- utf-8 -*-

import numpy as np

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float, line) for line in stringArr]
    return np.mat(dataArr)
