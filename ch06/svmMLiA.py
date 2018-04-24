# -*- coding: utf-8 -*-
import numpy as np
from time import sleep


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 随机选择一个不等于i的数j并返回
# i 代表alpha的下标，m代表alpha的数目
def selectJrand(i, m):
    j = i
    while i == j:
        j = int(np.random.uniform(0, m))
    return j


# 调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 分类集
    :param C: 常数
    :param toler: 容错率
    :param maxIter: 做大迭代次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros(m, 1))
    iter = 0
    while iter < maxIter:
        # 标记位，记录是否优化
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # 误差
            Ei = fXi - float(labelMat[i])
        if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
            j = selectJrand(i, m)
            fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
            Ej = fXj - float(labelMat[j])
            alphaIold = alphas[i].copy()
            alphaJold = alphas[j].copy()
            # 计算L和H，用于将alpha[j]调整到0到C之间
            if labelMat[i] != labelMat[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                print("L==H")
                continue
            # eta是alpha[j]的最优修改量, L的二阶导数
            eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
            if eta >= 0:
                print("eta>=0")
                continue
            # 更新
            alphas[j] -= labelMat[j]*(Ei - Ej)/eta
            alphas[j] = clipAlpha(alphas[j], H, L)
            if abs(alphas[j] - alphaJold) < 0.00001:
                print("j not moving enough")
                continue
            # 反向更新alpha[i]
            alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
            