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


def regLeaf(dataSet):#returns the value used for each leaf
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    # np.var:Returns the variance of the array elements
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    #assume dataSet is NumPy Mat   #choose the best split
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        # if the splitting hit a stop condition return val
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


