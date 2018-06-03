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
            minDist = np.inf
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


def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 整个数据集当成一个簇，计算其质心
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 创建只有一个质心的列表
    centList = [centroid0]
    for j in range(m):#calc initial Error
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
        while (len(centList) < k):
            # SSE (Sum of Squared Error,误差平方和）
            lowestSSE = np.inf
            for i in range(len(centList)):
                # 取得在当前簇中的所有数据点
                ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
                centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
                print(centroidMat)
                sseSplit = sum(splitClustAss[:, 1])#compare the SSE to the currrent minimum
                sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
                print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            # 上面应用簇为2的分类得到的index有0和1，要改成新的
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
            print('the bestCentToSplit is: ', bestCentToSplit)
            print('the len of bestClustAss is: ', len(bestClustAss))
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
            centList.append(bestNewCents[1, :].tolist()[0])
            # 赋予新的误差
            clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment


# import urllib
# import json
# def geoGrab(stAddress, city):
#     apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
#     params = {}
#     params['flags'] = 'J'#JSON return type
#     params['appid'] = 'aaa0VN6k'
#     params['location'] = '%s %s' % (stAddress, city)
#     url_params = urllib.parse.urlencode(params)
#     yahooApi = apiStem + url_params      #print url_params
#     print(yahooApi)
#     c = urllib.request.urlopen(yahooApi)
#     return json.loads(c.read())
#
#
# from time import sleep
# def massPlaceFind(fileName):
#     fw = open('places.txt', 'w')
#     for line in open(fileName).readlines():
#         line = line.strip()
#         lineArr = line.split('\t')
#         retDict = geoGrab(lineArr[1], lineArr[2])
#         if retDict['ResultSet']['Error'] == 0:
#             lat = float(retDict['ResultSet']['Results'][0]['latitude'])
#             lng = float(retDict['ResultSet']['Results'][0]['longitude'])
#             print("%s\t%f\t%f" % (lineArr[0], lat, lng))
#             fw.write('%s\t%f\t%f\n' % (line, lat, lng))
#         else:
#             print("error fetching")
#         sleep(1)
#     fw.close()
