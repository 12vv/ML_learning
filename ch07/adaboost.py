# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    # 特征数目
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 单层决策树生成
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 寻找最佳特征值
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 初始化最小错误率为正无穷
    minError = np.inf
    # 对每一列，即每一个特征
    for i in range(n):
        # 找最大最小值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 步长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历当前特征所有值
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                # 阈值，即划分类的依据
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化误判矩阵为1，正确判断的位置置为0
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# AdaBoost算法
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 初始化每个数据点的权重为1/m
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.zeros((m, 1))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # ??? 计算alpha
        alpha = float(0.5 * np.log((1.0-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # ??? 更新权重向量D用于下一轮循环
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 计算错误率，若为0的话使用break退出循环
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# 利用训练出来的多个分类器进行分类
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[0][i]['dim'],
                                 classifierArr[0][i]['thresh'], classifierArr[0][i]['ineq'])
        aggClassEst += classifierArr[0][i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


# 画 ROC 图
def plotROC(predStrengths, classLabels):
    # 光标位置
    cur = (1.0, 1.0)
    # 计算 AUC 用到的变量
    ySum = 0.0
    # 正类数量
    numPosClas = sum(np.array(classLabels) == 1.0)
    # y轴，x轴上的步进
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 排序索引，从小到大
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    print(sortedIndicies)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            # 降低真阳率
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # 绘制从cur到(cur[0]-delX,cur[1]-delY)的线
        # print(cur)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)



