# -*- coding: utf-8 -*-
from ch04 import bayes
from importlib import reload
import feedparser


reload(bayes)

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)

# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
#
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)

# bayes.spamTest()

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny = feedparser.parse('https://newyork.craigslist.org/search/mnh?format=rss')
sf = feedparser.parse('https://newyork.craigslist.org/search/brk?format=rss')
# ny['entries']
# print(ny)
# vocabList, pSF, pNY = bayes.localWords(ny, sf)
#
# print(len(vocabList))

bayes.getTopWords(ny, sf)


