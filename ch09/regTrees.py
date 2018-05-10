# -*- coding: utf-8 -*-
import numpy as np


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 元素类型转为浮点
        fltLine = np.map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    # print(dataSet[:, feature] > value)
    # print(np.nonzero(dataSet[:, feature] > value))
    # print(np.nonzero(dataSet[:, feature] > value)[0])
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

