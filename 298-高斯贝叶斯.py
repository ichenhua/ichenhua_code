# 1、导入并拆分数据集
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
x, y = digits.data, digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)

# 2、建模
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

score = gnb.score(x_test, y_test)
print(score) #0.827

# 3、探索建模结果

from sklearn.metrics import confusion_matrix as CM

# 查看预测结果的概率
proba = gnb.predict_proba(x_test)
print(proba.shape) # 返回的是每个样本，对应每个分类的概率

print(proba.sum(1)) # 每一行的概率之和都是1

# 查看混淆矩阵
print(CM(y_test, y_pred))



