# -*- coding: utf-8 -*-
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fitLine = list(map(np.float, curLine))
        dataMat.append(fitLine)
    return dataMat


# 计算欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    # 列，相当于数据集中数据点维度
    n = np.shape(dataSet)[1]
    # 质心矩阵
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """

    :param dataSet: dataSet
    :param k: the number of clusters
    :param distMeas: (optional) a function to use as the distance metric
    :param createCent: (optional) a function to create the initial centroids
    :return: centroids and cluster assignments
    """
    # 数据点个数
    m = np.shape(dataSet)[0]
    # 第一列存放簇的索引值，第二列存放误差，误差是当前点到质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:# 直到没有点的簇更新了就退出循环
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -1
            for j in range(k):# 寻找与其距离最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        # 重新计算质心
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]#get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment