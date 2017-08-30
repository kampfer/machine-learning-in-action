# -*- coding: utf-8 -*-

import fpGrowth

def test():
    simpDat = fpGrowth.loadSimpDat()
    initSet = fpGrowth.createInitSet(simpDat)
    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)

test()
