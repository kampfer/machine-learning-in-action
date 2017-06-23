import pandas as pd
import kNN

data = pd.read_table('datingTestSet2.txt', names=['a', 'b', 'c', 'd'])
normData, ranges, minVals = kNN.autoNorm(data.iloc[:,:-1])
datingLabels = data.d
numTestVecs = int(0.1 * normData.shape[0])
errorCount = 0
for i in range(numTestVecs):
    classifierResult = kNN.classify0(normData.iloc[i].values, normData.iloc[numTestVecs:].values, datingLabels.iloc[numTestVecs:].values, 3)
    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, data.d[i])
    if (classifierResult != data.d[i]):
        errorCount += 1;
print "the total error rate is: %f" % (errorCount / numTestVecs)
