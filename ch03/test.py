import trees
import treePlotter

myData, labels = trees.createDataSet()
myTree = trees.createTree(myData, labels[:])
print trees.classify(myTree, labels, [1,0]), trees.classify(myTree, labels, [1,1])

# fr = open('lenses.txt')
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# lenseTree = trees.createTree(lenses, lenseLabels)
# treePlotter.createPlot(lenseTree)
