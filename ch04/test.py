import bayes

postList, classList = bayes.loadDataSet()
myVocabList = bayes.createVocabList(postList)
trainMat = []
for post in postList:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, post))
p0Vec, p1Vec, pAb = bayes.trainNB0(trainMat, classList)
print p0Vec, p1Vec, pAb
