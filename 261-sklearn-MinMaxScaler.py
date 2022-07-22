# Sklearn数据归一化MinMaxScaler

# 归一化（Normalization）和数据标准化（Standardization），是数据无量纲化的两大常用方法。归一化的方法是先按最小值中心化之后，再按极差（最大值-最小值）缩放，即数据先移动最小值个单元，在缩放使其收敛于[0,1]之间。归一化后的数据服从正态分布。

# 公式表示：x' = (x-min)/(max-min)

# Numpy实现归一化
import numpy as np

x = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# 归一化
x_norm = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
print(x_norm)

# 归一化逆转
x_inv = x_norm * (x.max(axis=0) - x.min(axis=0)) + x.min(axis=0)
print(x_inv)

# Sklearn实现归一化
# 在Sklearn当中，可以使用 sklearn.preprocessing.MinMaxScaler 来实现归一化，MinMaxScaler 有一个重要参数 feature_range，控制我们希望把数据压缩到的范围，默认[0，1]。

from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = MinMaxScaler()
scaler.fit(data)  # 这里本质是生成min和max
x_norm = scaler.transform(data)  # 通过接口导出数据
print(x_norm)

# 训练和导出一步到位
x_norm = scaler.fit_transform(data)
print(x_norm)

# 归一化结果逆转
x_inv = scaler.inverse_transform(x_norm)
print(x_inv)

# 通过参数 feature_range 将数据归一化到[0,1]以外的范围
scaler = MinMaxScaler(feature_range=[5,10])
x_norm = scaler.fit_transform(data)
print(x_norm)

# 当x中特征数量较多时，fit可能会报错，此时可以使用 partial_fit 作为训练接口
scaler = scaler.partial_fit(data)
