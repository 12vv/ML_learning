# -*- coding: utf-8 -*-
from ch02 import KNN
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np


group, labels = KNN.createDataSet()

# print(group, labels)


print(KNN.classify0([0, 0], group, labels, 3))

reload(KNN)
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels[0:20])

# 散点图
fig = plt.figure()
ax = fig.add_subplot(111)
print(15.0*np.array(datingLabels))
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), c=15.0*np.array(datingLabels))
plt.show()

# reload(KNN)
# normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
# print(normMat, minVals)
#
# KNN.datingClassTest()

# testVector = KNN.img2vector('testDigits/0_13.txt')
# print(testVector[0, 0:31])

KNN.handwritinClassTest()
