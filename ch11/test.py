# -*- coding: utf-8 -*-

import apriori

def test1():
    dataSet = apriori.loadDataSet()
    C1 = apriori.createC1(dataSet)
    L, supportData = apriori.apriori(dataSet)
    print(L)

test1()
