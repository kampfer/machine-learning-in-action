# -*- coding: utf-8 -*-

import bayes
import re
import numpy as np

def testingNB():
    postList, classList = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(postList)
    trainMat = []
    for post in postList:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, post))
    p0V, p1V, pAb = bayes.trainNB0(trainMat, classList)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = bayes.setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = bayes.setOfWords2Vec(myVocabList, testEntry)
    print testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb)

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        # trainMat.append(bayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        # wordVector = bayes.bagOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print 'classification error',  docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)

# testingNB()
spamTest()
