# sklearn实现KNN算法及KDtree算法优化
# http://www.ichenhua.cn/read/232
# @author chenhua

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 定义数据
kiss = [104, 100, 81, 10, 5, 2]
fight = [3, 2, 1, 101, 99, 98]
labels = [1, 1, 1, 2, 2, 2]

test = [90, 18]

X_train = np.array([kiss, fight]).T
y_train = np.array(labels)

# 调包训练
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
knn.fit(X_train, y_train)

test_X = np.array([test])
pred_y = knn.predict(test_X)
print(pred_y)
