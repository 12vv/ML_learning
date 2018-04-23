# -*- coding: utf-8 -*-
from ch05 import logRegres
from importlib import reload
import numpy as np


reload(logRegres)
dataArr, labelMat = logRegres.loadDataSet()
w = logRegres.gradAscent(dataArr, labelMat)
# print(w)

# w.getA()返回ndarray方便取值
# logRegres.plotBestFit(w.getA())

weights = logRegres.stocGradAscent1(np.array(dataArr), labelMat)
logRegres.plotBestFit(weights)

