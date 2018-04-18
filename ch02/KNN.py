# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator
from os import listdir


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类器
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


# 读取数据文件
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


# 归一化特征值
# new_val = (new_val - min) / (max - min)
def autoNorm(dataSet):
    # 取每一列的最小值，最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 原数据集中每一个元素减去此元素所在列的最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 新的数据集每个元素除以当前列最大值和最小值之差
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试分类器
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 取数据集的10%
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
        print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


# 处理数字图像
# 将32*32转换成1*1024向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect


# 数字识别
def handwritinClassTest():
    hwLabels = []
    # listdir获取目录下的子目录
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名中提出数字（文件名类似于：0_0.txt，0_2.txt）
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/' + fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
