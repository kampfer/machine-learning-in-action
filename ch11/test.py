# -*- coding: utf-8 -*-

import apriori

def test1():
    dataSet = apriori.loadDataSet()
    C1 = apriori.createC1(dataSet)
    L, supportData = apriori.apriori(dataSet, minSupport=0.5)
    rules = apriori.generateRules(L, supportData)
    print(rules)

test1()
