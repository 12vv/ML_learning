# -*- coding: utf-8 -*-
from ch07 import adaboost
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload


reload(adaboost)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# dataMat, classMat = adaboost.loadSimpData()
# # 根据分类赋予绘制不同颜色的点
# colors = ['red' if l == 1 else 'green' for l in classMat]
# print(type(np.array(classMat)))
# ax.scatter(np.array(dataMat[:, 0]), np.array(dataMat[:, 1]), c=np.array(colors))
# plt.show()

datArr, labelArr = adaboost.loadSimpData()
# classifierArr = adaboost.adaBoostTrainDS(datArr, labelArr, 30)
# adaboost.adaClassify([0, 0], classifierArr)
classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr,labelArr,10)
adaboost.plotROC(aggClassEst.T, labelArr)

