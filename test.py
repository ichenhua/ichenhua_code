
import numpy as np
import torch

def glorot(shape):
    return torch.randn(*shape)
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return torch.FloatTensor(shape[0], shape[1]).uniform_(-init_range, init_range)

print(glorot((3,4)))


exit()

import torch
import torch.nn as nn

model = nn.LSTM(100, 64, num_layers=2,  batch_first=True, bidirectional=True)

x = torch.rand(10, 31, 100)

print(model(x)[1][0].shape)

exit()



def fn1():
    for i in range(4):
        print(1)
        yield i

class C():
    def __iter__(self):
        for i in fn1():
            yield i

c = C()

for i in c:
    print(i)
    exit()

exit()


import pandas as pd

df = pd.DataFrame({'a':1, 'b':2})

print(df)


exit()


# 澳大利亚观测点城市气候区域划分

# 本文是澳大利亚天气预测项目的前置数据处理环节，在大项目中需要将观测点所在城市转化为气候区域，以方便探究气候区域与天气的关系。
# 但从公开数据中，只能查到主要城市的气候区域，所以处理思路是，通过观测点和主要城市的经纬度，计算出实际距离，然后近似找到观测点的气候区域。


# 1、加载数据

import pandas as pd

# 样本城市经纬度
sample_city_ll = pd.read_csv('./datas/sample_city_ll.csv', index_col=0)
# 主要城市经纬度
city_ll = pd.read_csv('./datas/city_ll.csv', index_col=0)
# 主要城市气候区域
city_climate = pd.read_csv('./datas/city_climate.csv', index_col=0)


# 2、使用geopy库按经纬度计算两点距离

from geopy.distance import geodesic

sample_df = pd.DataFrame(index=sample_city_ll['City'])
# 遍历获取sample_city信息
for idx, sample_row in sample_city_ll.iterrows():
    sample_row = dict(sample_row)
    sample_city = sample_row['City']
    sample_lt = sample_row['Latitude'].strip('°')
    sample_lg = sample_row['Longitude'].strip('°')
    # 遍历获取city信息
    dists = []
    for idx, row in city_ll.iterrows():
        row = dict(row)
        city = row['City']
        lt = row['Latitude'].strip('°')
        lg = row['Longitude'].strip('°')
        # 计算距离
        dists.append([geodesic((sample_lt, sample_lg), (lt, lg)).km, city])
    # 获取距离最小值对应的城市
    _, city = min(dists)
    # 查找最近城市对应的气候区域，并填充dataframe
    climate = city_climate.loc[city, 'Climate']
    sample_df.loc[sample_city, 'Climate'] = climate

sample_df.to_csv('./datas/sample_city_climate.csv')
print(sample_df.shape)

exit()

# https://www.cnblogs.com/shenxiaolin/p/8854197.html

import numpy as np
import matplotlib.pyplot as plt

x, y = np.mgrid[1:3:3j, 1:2:2j]
print(x.flatten())
print(y.flatten())

plt.scatter(x.flatten(), y.flatten())
plt.show()
