# -*- coding: utf-8 -*-
from ch10 import kMeans
import numpy as np
from importlib import reload

# datMat = np.mat(kMeans.loadDataSet('testSet.txt'))

# print(np.min(datMat[:, 0]))
# print(np.max(datMat[:, 1]))
# print(kMeans.randCent(datMat, 2))
# print(kMeans.distEclud(datMat[0], datMat[1]))

# myCentroids, clustAssing = kMeans.kMeans(datMat, 4)
# print(myCentroids)
reload(kMeans)
datMat3 = np.mat(kMeans.loadDataSet('testSet2.txt'))
centList, myNewAssments = kMeans.biKmeans(datMat3, 3)
print(centList)
# geoResults = kMeans.geoGrab('1 VA Center', 'Augusta, ME')