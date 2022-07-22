# 手写AI算法之最小二乘法求解多元线性回归

# 前面文章中，介绍了一元线性回归的求解，其关键点在于系数w和截距b的求解。但如果是多元特征，需要求解的变量就是多维的，之前的方法就会异常复杂。
# 所以对于多元特征问题，求解的思路是将向量聚合，用矩阵的方式求解。

# 以下用经典的波士顿房价预测问题，来介绍多元线性回归问题代码求解方式。

# CRIM	城镇人均犯罪率	float
# ZN	住宅用地超过 25000 sq.ft. 的比例	float
# INDUS	城镇非零售商用土地的比例	float
# CHAS	查理斯河空变量（如果边界是河流，则为1；否则为0）	int
# NOX	一氧化氮浓度	float
# RM	住宅平均房间数	float
# AGE	1940 年之前建成的自用房屋比例	float
# DIS	到波士顿五个中心区域的加权距离	float
# RAD	辐射性公路的接近指数	float
# TAX	每 10000 美元的全值财产税率	float
# PTRATIO	城镇师生比例	float
# B	1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例	float
# LSTAT	人口中地位低下者的比例	float
# MEDV	自住房的平均房价，以千美元计	float

# 代码示例

# 1、加载数据并处理
import numpy as np
from matplotlib import pyplot as plt

ds = np.loadtxt('./datas/boston_housing.data')
x_arr = ds[:, :-1]
y_arr = ds[:, -1].reshape(-1, 1)

X = np.mat(np.hstack([np.ones((x_arr.shape[0], 1)), x_arr]))
Y = np.mat(y_arr)


# 2、参数求解
def fit(X, Y):
    theta = (X.T * X).I * X.T * Y
    return theta


theta = fit(X, Y)
y_hat = X * theta

# print('theta：', theta)
# print(y_hat.shape)

# 3、可视化
# x = range(len(y_hat))
# plt.scatter(x, y_arr)
# plt.plot(x, y_hat, c='r')
# plt.show()

# 4、模型预测
test_x = np.mat([[1, 2.27346, 0.00, 18.580, 1, 0.7050, 6.2500, 92.60, 1.7984, 5, 403.0, 14.70, 348.92, 6.50]])
pred_y = test_x * theta
print('预测值：', pred_y[0, 0])
