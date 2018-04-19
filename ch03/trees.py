# -*- coding: utf-8 -*-
from importlib import reload
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # featVec[-1]表示当前行最后一列的值，做分类标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        # 计算每一种分类的概率
        prob = float(labelCounts[key]) / numEntries
        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 取本行除了axis之外的其他列再拼起来 chop out axis used for splitting
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的划分方式
def chooseBestFeatureToSplit(dataSet):
    # 特征值数量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # featList中存放每个样本中此特征的值
        featList = [example[i] for example in dataSet]
        # 使用set()得到list中唯一元素，去掉值重复的元素
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # 用此特征划分得到的新信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 与目前最优的增益对比，如果大则更新增益以及此特征值
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



