

# 构建测试数据
from sklearn.datasets import make_moons
data, target = make_moons(n_samples=2000, noise=0.3, random_state=42)

# 数据可视化
from matplotlib import pyplot as plt
plt.scatter(data[:,0], data[:,1], c=target)
plt.show()

# 拆分数据集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

# 训练模型
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# 测试准确率
score_train = classifier.score(x_train, y_train)
print('score_train:', score_train)
score_test = classifier.score(x_test, y_test)
print('score_test:', score_test)


