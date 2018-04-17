# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """

    :param inX: 输入的将要分类的向量
    :param dataSet: 训练样本
    :param labels: 标签向量
    :param k: 选择最邻近邻居数目
    :return:
    """
    dataSetSize = dataSet.shape[0]
    # 计算距离
    # tile使inx行复制dataSetSize次，列方向不进行复制(参数1)，再与dataSet对应元素相减
    # 即计算欧拉距离中的(xA0-xB0),(xA1-xB1)...
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 计算 (xA0-xB0)^2,(xA1-xB1)^2
    sqDiffMat = diffMat**2
    # 每一行相加，即 (xA0-xB0)^2 + (xA1-xB1)^2
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号，得出此点与样本中所有点的欧拉距离
    distances = sqDistances**0.5
    # 根据距离排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 通过iteritems函数将classCount转化为可迭代对象再排序。
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    # 取得文件有多少行
    numberOfLines = len(arrayOlines)
    # 创建矩阵，3为数据样本参数
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件到数据列表
    for line in arrayOlines:
        # 移除首尾字符(默认空格)
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 索引为-1代表取最后一列元素，这里最后一列为所分类别
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

