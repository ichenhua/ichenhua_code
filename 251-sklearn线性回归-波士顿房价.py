# sklearn求解多元线性回归问题

# 前面文章中，介绍了推导公式和手写代码的形式，来求解多元线性回归问题。但在真实项目中，一般都会使用调库的方式来完成任务。
# 以下依然以波士顿房价预测需求为例，来介绍使用sklearn求解多元线性回归问题的方法。

# 参考文档：https://blog.csdn.net/weixin_43094965/article/details/121090020

# 代码示例

# 1、加载数据，并拆分数据集
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, train_test_split

ds = load_boston()
x = ds.data
y = ds.target

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 2、模型训练
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)

# 3、模型预测
y_hat = reg.predict(x_test)
# 平方损失
score = reg.score(x_test, y_test)
print(y_hat)
print(score)  #0.783

# 4、预测结果可视化
from matplotlib import pyplot as plt

plt.scatter(range(len(y_test)), sorted(y_test), label='y_test')
plt.scatter(range(len(y_test)), sorted(y_hat), label='y_hat')
plt.legend()
plt.show()

# 5、探索模型
# 各特征对应系数
print(list(zip(ds.feature_names, reg.coef_)))
# 截距项
print(reg.intercept_)

# 6、疑惑点
# 交叉验证时，R2跨度很大
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, x, y, cv=5)
print(scores)

#[0.63919994  0.71386698  0.58702344  0.07923081 -0.25294154]
