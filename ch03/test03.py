# -*- coding: utf-8 -*-
from ch03 import trees
from importlib import reload

reload(trees)
myDat, labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))

# print(trees.splitDataSet(myDat, 0, 1))
# print(trees.splitDataSet(myDat, 0, 0))

print(trees.chooseBestFeatureToSplit(myDat))
