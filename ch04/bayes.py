# -*- coding: utf-8 -*-
from numpy import *
import feedparser


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return postingList, classVec


# 创建不重复词表
def createVocabList(dataSet):
    # 先创建一个空集合
    vocabSet = set([])
    for document in dataSet:
        # 两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 返回inputSet向量中是否出现vocabList单词
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 词带(bag)与词集(set)不同在于，词带中元素可以多次出现
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    """

    :param trainMatrix: 文档矩阵，每个文档的长度可能不同，文档中元素为单词
    :param trainCategory: 文档类型分类 0/1
    :return: 两个向量，一个概率pAbusive
    """
    numTrainDocs = len(trainMatrix)
    # 每个文档中的词数
    numWords = len(trainMatrix[0])
    # 被划分为侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 避免其中一个概率为0 结果(乘积)就为0
    # 把所有词出现数初始化为1，分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 经过上面的循环，Denom为各个分类单词的总数(+2,因为初始化为2)，Num为单词表中每个单词出现的总数(+1)，向量
    # p1Num/p1Denom表示单词表中每个单词出现在1分类下的概率
    # 取对数避免下溢出，许多很小的数相乘浮点数舍入导致错误，采用自然对数处理不会有损失
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify: 将要进行分类的向量
    :param p0Vec: trainNB0中的 p0Vect
    :param p1Vec: trainNB0中的 p1Vect
    :param pClass1: trainNB0中的 pAbusive
    :return: 所分类别
    """
    # 使用 * 乘符号，在numpy中表示对应位置元素相乘，累加相当于p(w1|1)+p(w2|1)+...
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 将文本转换为词列表
def textParse(bigString):
    """

    :param bigString: 字符串文本
    :return: 单词列表
    """
    import re
    # 加上r表示原生字符串，\W匹配所有非字母，数字或下划线
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 垃圾邮件分类
def spamTest():
    docList=[]
    classList = []
    fullText =[]
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # spam中的邮件是垃圾邮件，分类为1
        classList.append(1)
    for i in range(1, 26):
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 构建不重复词表
    vocabList = createVocabList(docList)
    # 整数列表
    trainingSet = list(range(50))
    # 创建测试集，随机抽取10个作为测试集
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # 训练
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 对剩下的(被当做测试集)进行分类
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))


# 遍历vocabList中的每个单词在fullText的次数并排序
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# 类似于spamTest
def localWords(feed1, feed0):
    """

    :param feed1: RSS源 1
    :param feed0: RSS源 2
    :return:
    """
    docList=[]
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        # NY is class 1 (源1)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词表
    vocabList = createVocabList(docList)
    # 移除出现频率最高的30个词
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        # pairW 是 ('with', 26) 这种形式(元组？)，pairW[0]则表示 'with'
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # ????? 乘2是为什么？
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V



def getTopWords(ny,sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    # 按照每个元素的第2个元素(每个元素是一个元组)进行排序，则这个词出现的次数
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
