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


# 找出出现次数最多的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 构造树，其中参数labels只是为了给出数据明确含义
def createTree(dataSet, labels):
    # 创建包含所有分类的list
    classList = [example[-1] for example in dataSet]
    # 如果classList[0]在classList中出现的次数==classList的长度，就表示分类统一(就这一种分类)，无需再分，直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 使用完了所有特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    # python 2
    # firstStr = inputTree.keys()[0]
    # 下面一行是 python3 写法
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 把特征值名称转换为索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# 使用pickle序列化对象，之后不用每次都构造决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
