# -*- coding: utf-8 -*-

import apriori

def test1():
    dataSet = apriori.loadDataSet()
    C1 = apriori.createC1(dataSet)
    L, supportData = apriori.apriori(dataSet, 0.7)
    print(L)
    print(supportData)

test1()
