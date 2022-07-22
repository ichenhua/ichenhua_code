# Sklearn数据标准化StandardScaler

# 前一篇文章讲到数据归一化，本文继续讲解无量纲化的第二种方法，数据标准化。其过程是先将数据按均值中心化后，再按标准差缩放，得到的数据服从均值为0，标准差为1的标准正态分布。

# 公式表示：x' = (x-μ)/δ

# Numpy实现标准化
import numpy as np

x = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# 标准化
x_std = (x-x.mean(axis=0)) / x.std(axis=0)
print(x_std)

# 标准化逆转
x_inv = x_std * x.std(axis=0) + x.mean(axis=0)
print(x_inv)


# Sklearn实现标准化
from sklearn.preprocessing import StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = StandardScaler()
scaler.fit(data)  # 计算均值和方差
print(scaler.mean_)  # 均值
print(scaler.var_)  # 方差

x_std = scaler.transform(data)
print(x_std)

# 训练和导出一步到位
x_std = scaler.fit_transform(data)
print(x_std)

# 标准化逆转
x_inv = scaler.inverse_transform(x_std)
print(x_inv)

