# -*- coding: utf-8 -*-
from ch03 import trees
from ch03 import treePlotter
from importlib import reload

reload(trees)
myDat, labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))

# print(trees.splitDataSet(myDat, 0, 1))
# print(trees.splitDataSet(myDat, 0, 0))

# print(trees.chooseBestFeatureToSplit(myDat))


# myTree = trees.createTree(myDat, labels)
# print(myTree)

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTrees = trees.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTrees)
