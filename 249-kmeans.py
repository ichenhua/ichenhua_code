import random
from matplotlib import axis, pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs


class KMeans():

    def __init__(self, k, iter_max):
        self.k = k
        self.iter_max = iter_max
        self.iter_num = 0

    def fit(self, X):
        self.X = X
        # 随机创建中心点
        ids = random.sample(range(len(self.X)), self.k)
        self.centers = self.X[ids]
        while True:
            r = self._classify()
            if r:
                return self
            self._update_centers()

    def _classify(self):
        # 按中心点分类
        self.classes = [[] for i in range(self.k)]
        self.labels = []
        sum_dist = 0
        for x in self.X:
            dist = ((np.tile(x, (self.k, 1)) - self.centers)**2).sum(axis=1)
            id = np.argsort(dist)[0]
            # 添加labels
            self.labels.append(id)
            # 累加距离
            sum_dist += dist[id]
            # 归类
            self.classes[id].append(x)
        self.sum_dist = sum_dist
        self.classes = np.array(self.classes)
        self.labels = np.array(self.labels)
        if abs(self.sum_dist - sum_dist) < 0.1:
            return True
        else:
            self.sum_dist = sum_dist

    def _update_centers(self):
        # 更新中心点
        centers = []
        for item in self.classes:
            item = np.array(item)
            centers.append(item.mean(axis=0))
        self.centers = centers


if __name__ == '__main__':
    X, _ = make_blobs(200, 2, centers=[[2, 3], [6, 8]])
    kmeans = KMeans(2, 10).fit(X)

    colors = ['red', 'blue', 'green', 'yellow']
    for x, l in zip(X, kmeans.labels):
        plt.scatter(x[0], x[1], c=colors[l])
    plt.show()