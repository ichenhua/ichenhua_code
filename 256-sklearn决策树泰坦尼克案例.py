# Sklearn决策树泰坦尼克号幸存者预测

# 泰坦尼克号的沉没是历史上最严重的沉船事件之一。1912年4月15日，在首次航行期间，泰坦尼克号撞上冰山后沉没，2224名乘客和机组人员中有1502人遇难。这场轰动的悲剧震撼了国际社会，并导致了更好的船舶安全条例。
# 海难导致生命损失的原因之一是没有足够的救生艇给乘客和机组人员。虽然幸存下来的运气有一些因素，但一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。最惨的是下流社会的男人，至少在西方社会是这样的。同时也看到了人道的光辉。
# 在这个挑战中，我们要求你完成对哪些人可能生存的分析。特别是，我们要求您运用机器学习的工具来预测哪些乘客幸免于难。

# PassengerId：乘客ID
# Survived：是否获救，1-获救,0-没有获救
# Pclass：乘客等级，1-Upper，2-Middle，3-Lower
# Name：乘客姓名
# Sex：性别
# Age：年龄
# SibSp：乘客在船上的配偶数量或兄弟姐妹数量
# Parch：乘客在船上的父母或子女数量
# Ticket：船票信息
# Fare：票价
# Cabin：是否住在独立的房间，1-是，0-否（有缺失）
# Embarked：登船港口 (C = Cherbourg, Q = Queenstown, S = Southampton)

# 1、导入数据、熟悉数据
import pandas as pd

data = pd.read_csv('./datas/titanic.csv')
print(data.head())
data.info()

# 2、数据预处理
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
# 填补缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())
data.dropna(inplace=True)

# 二分类数值转换
data['Sex'] = (data['Sex'] == 'male').astype('int')
# 三分类数值转化
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))

# 3、拆分数据集
x = data.loc[:, data.columns != 'Survived']
y = data.loc[:, data.columns == 'Survived']

from sklearn.model_selection import GridSearchCV, train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 重置索引
for i in [x_train, x_test, y_train, y_test]:
    i.reset_index(drop=True, inplace=True)

# 4、模型训练
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
score_train = clf.score(x_train, y_train)
print('score_train:', score_train)
score_test = clf.score(x_test, y_test)
print('score_test:', score_test)

# 交叉验证
from sklearn.model_selection import cross_val_score

score = cross_val_score(clf, x, y, cv=10).mean()
print(score)

# 5、最大深度拟合曲线
import matplotlib.pyplot as plt

score_train_list = []
score_test_list = []
for i in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(x_train, y_train)
    score_train = clf.score(x_train, y_train)
    score_train_list.append(score_train)
    score_test = clf.score(x_test, y_test)
    score_test_list.append(score_test)

plt.plot(range(1, 11), score_train_list, label='score_train')
plt.plot(range(1, 11), score_test_list, label='score_test')
plt.legend()
plt.show()

# 6、网格搜索调整参数
from sklearn.model_selection import GridSearchCV
import numpy as np

clf = DecisionTreeClassifier()
params = {
    'max_depth': range(1, 11), 
    'min_samples_leaf': np.linspace(0.01, 0.99, 10)
}
gs = GridSearchCV(clf, params, cv=10)
gs.fit(x, y)
print(gs.best_params_)
print(gs.best_score_)
