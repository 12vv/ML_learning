# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    # readlines返回一个list方便遍历
    for line in fr.readlines():
        lineArr = line.strip().split()
        # dataMat有三列，第一列为了方便加上的，默认x0 = 1.0 第二列和第三列为x1和x2
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    # convert to NumPy matrix
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # dataMatrix * weights是矩阵相乘
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        # ???按照差值调整weights   ???
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 绘制拟合直线
def plotBestFit(weights):
    dataMat, labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        # 分类为1
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 拟合曲线：0 = w0*x0+w1*x1+w2*x2 ， 则 x2 = (-w0*x0-w1*x1)/w2
    # 此处x2由y表示
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 对应位置相乘再相加，同矩阵乘法
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 注意，这里error是数值不是向量
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进梯度上升
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次调整alpha，使其随着迭代次数不断减小，加一个常数项，让它不会减小到0
            alpha = 4/(1.0+j+i) + 0.0001
            # 随机选取样本更新w
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

