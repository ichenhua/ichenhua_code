# 1、生成数据集

import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.RandomState(24)
x = rnd.uniform(-3, 3, size=100)
y = np.sin(x) + rnd.normal(size=len(x)) / 3

# plt.scatter(x, y)
# plt.show()

# sklearn只接受二维数据，先转化
x = x.reshape(-1, 1)

# 2、使用原始数据

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(x, y)
print(linear.score(x, y))  #0.500

# 创建连续变量进行预测
line = np.linspace(-3, 3, 100).reshape(-1,1)
line_pred = linear.predict(line)

# plt.plot(line, line_pred, c='red')
# plt.scatter(x, y)
# plt.show()

# R2分数0.5，拟合的是一条直线，看不出sin函数的变化趋势，效果很差。

# 3、多项式回归

from sklearn.preprocessing import PolynomialFeatures

# 创建3次项，保留交叉项（两个以上的特征时才有意义），不要截距项（线性回归模型中会添加）
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
x_ = poly.fit_transform(x)

# 再做线性拟合
linear = LinearRegression()
linear.fit(x_, y)
print(linear.score(x_, y))

# 创建连续变量
line = np.linspace(-3, 3, 100).reshape(-1, 1)
line_ = poly.fit_transform(line)
y_pred_ = linear.predict(line_)

plt.scatter(x, y)
plt.plot(line, y_pred_, c='red')
plt.show()

# 由图可见，将特征扩展到三次项后，基本能完全拟合出sin曲线了。
