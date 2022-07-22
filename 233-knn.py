# 手写AI算法之KNN近邻算法
# http://www.ichenhua.cn/read/233
# @author chenhua

import numpy as np
from collections import Counter

class Knn():
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def perdict(self, X):
        res = []
        for x in X:
            # 求欧式距离
            dists = [((x-x_t)**2).sum()**0.5 for x_t in self.X_train]
            idxs = np.argsort(dists)
            # 找到前k个id对应的y值
            ls = self.y_train[idxs[:self.k]]
            # 统计数量，取最多
            res.append(Counter(ls).most_common()[0][0])
        return np.array(res)


# 定义数据
kiss = [104, 100, 81, 10, 5, 2]
fight = [3, 2, 1, 101, 99, 98]
labels = [1, 1, 1, 2, 2, 2]

test = [90, 18]

# 模型实例化
knn = Knn(5) # 一般用奇数
X_train = np.array([kiss, fight]).T
y_train = np.array(labels)
knn.fit(X_train, y_train)

# 模型预测
X_test = np.array([test])
res = knn.perdict(X_test)
print(res)


