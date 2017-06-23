import kNN
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table('datingTestSet2.txt', names=['a', 'b', 'c', 'd'])
fig = plt.figure()
ax = fig.add_subplot(311)
ax.scatter(data.a, data.b, 15.0*data.d, 15.0*data.d)
ax2 = fig.add_subplot(312)
ax2.scatter(data.b, data.c, 15.0*data.d, 15.0*data.d)
ax3 = fig.add_subplot(313)
ax3.scatter(data.a, data.c, 15.0*data.d, 15.0*data.d)
plt.show()
