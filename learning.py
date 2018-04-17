# -*- coding: utf-8 -*-
import KNN
import numpy as np
from importlib import reload
import matplotlib
import matplotlib.pyplot as plt

group, labels = KNN.createDataSet()

# print(group, labels)


print(KNN.classify0([0, 0], group, labels, 3))

reload(KNN)
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])

# 散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()

