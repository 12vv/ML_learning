# -*- coding: utf-8 -*-
from ch11 import apriori
import numpy as np
from importlib import reload


dataSet = apriori.loadDataSet()
# print(dataSet)

# C1 = apriori.createC1(dataSet)
# print(C1)
#
# D = list(map(set, dataSet))
# print(D)
#
# L1, suppData0 = apriori.scanD(D, C1, 0.5)
# print(L1, suppData0)

# L, suppData = apriori.apriori(dataSet)
# print(L)

L, suppData = apriori.apriori(dataSet, minSupport=0.5)
rules = apriori.generateRules(L, suppData, minConf=0.7)
print(rules)

