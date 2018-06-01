# -*- coding: utf-8 -*-
from ch10 import kMeans
import numpy as np


datMat = np.mat(kMeans.loadDataSet('testSet.txt'))

# print(np.min(datMat[:, 0]))
# print(np.max(datMat[:, 1]))
# print(kMeans.randCent(datMat, 2))
# print(kMeans.distEclud(datMat[0], datMat[1]))

myCentroids, clustAssing = kMeans.kMeans(datMat, 4)

print(myCentroids)
