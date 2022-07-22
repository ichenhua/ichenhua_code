# 手写AI算法之最小二乘法求解一元线性回归

# 参考文档
# https://blog.csdn.net/qq_34740277/article/details/107201263
# https://blog.csdn.net/xiewenrui1996/article/details/107418803

# 线性回归是利用数理统计中的回归分析，来确定两种或两种以上属性间相互依赖的定量关系的一种统计分析方法。
# 本文以学习时长和成绩之间的关系，学习时长越长，成绩越高，来阐释一元线性关系。

# 线性回归模型
# 线性回归（linear regression）是一种线性模型，它假设输入变量x和单个输出变量y之间存在线性关系。
# 公式表示：y = wx + b
# 线性方程求解，只需要两组xy参数，就能求出对应的w和b。但现实场景中，数据会多于两组，而且并不刚好落在同一条直线上，
# 线性模型（linear model）试图学得一组参数值，来拟合变量之间的关系。

# 最小二乘法和线性回归求解
# 最小二乘法的主要思想是，选择未知参数，使得预测值和观测值之差的平方和最小。

# 代码示例

# 1、模拟数据
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(1, 10, 100)
y = 6 * x + 20 + 10 * np.random.randn(100)

plt.scatter(x,y)
plt.show()


# 2、参数求解
def fit(x, y):
    M = len(x)
    x_bar = x.mean()
    w = (y * (x - x_bar)).sum() / ((x**2).sum() - M * ((x_bar**2).sum()))
    b = (y - w * x).sum() / M
    return w, b

w, b = fit(x, y)
y_hat = w * x + b

print('w：', w)
print('b：', b)

plt.scatter(x, y)
plt.plot(x, y_hat, c='r')
plt.show()


# 3、计算损失函数
def loss(w, b, x, y):
    return (((w * x + b) - y)**2).sum() / len(x)

l = loss(w, b, x, y)
print('loss:', l)
