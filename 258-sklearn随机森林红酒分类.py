# Sklearn随机森林实现红酒分类

# 随机森林是最简单的集成学习算法，其核心是两个随机加多棵CART树，即从样本集中有放回地随机选择n个样本，再从所有属性中随机选择k个属性，
# 重复以上步骤，来建立m棵决策树，最后通过投票表决，决定数据属于哪一类别。本文依然以Sklearn数据为例，来对比随机森林和决策树的分类效果。

# 1、导入并拆分数据集
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)



# 2、建立随机森林和决策树模型

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)
score_c = clf.score(x_test, y_test)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)
score_r = rfc.score(x_test, y_test)

print('single tree:', score_c, 'random forest:', score_r)




# 3、交叉验证效果对比
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier()
score_c = cross_val_score(clf, wine.data, wine.target, cv=10)

rfc = RandomForestClassifier()
score_r = cross_val_score(rfc, wine.data, wine.target, cv=10)

plt.plot(range(1,11), score_c, label='decision tree')
plt.plot(range(1,11), score_r, label='random forest')
plt.legend()
plt.show()



