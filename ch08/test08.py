# -*- coding: utf-8 -*-
from ch08 import regression
import numpy as np
import matplotlib.pyplot as plt


xArr, yArr = regression.loadDataSet('ex0.txt')
# print(xArr[0:2])
ws = regression.standRegres(xArr, yArr)
# print(ws)

xMat = np.mat(xArr)
yMat = np.mat(yArr)

# 预测值yhat
yHat = xMat * ws
# 通过 corrcoef 计算相关系数，即预测值和真实值的关系
print(np.corrcoef(yHat.T, yMat))

fig = plt.figure()
ax = fig.add_subplot(111)

# print(xMat[:, 1])
# print(xMat[:, 1].flatten())
# print(xMat[:, 1].flatten().A[0])
# print(yMat.T)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# 先对点排序再进行绘制
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()



