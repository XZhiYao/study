import numpy as np

top_k = 4

Array = np.array([2, 4, 6, 78, 4, 1, 64, 3, 5, 67, 3, 2, 233])
# print('argsort:', Array.argsort())
# argsort（）函数将数组x中的元素从小到大排序，并且取出他们对应的索引，然后返回
# Array.argsort():[ 5  0 11  7 10  1  4  8  2  6  9  3 12]
index = Array.argsort()[::-1][0:top_k]
# Array.argsort()[:-1] --> [ 5  0 11  7 10  1  4  8  2  6  9  3] （最后一位没取到）
# Array.argsort()[::-1] --> [12  3  9  6  2  8  4  1 10  7 11  0  5] （所有的位都取到了）
# Array.argsort()[::-1][0:top_k] --> [12  3  9  6]
print(index)



