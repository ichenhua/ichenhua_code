import pandas as pd
import numpy as np

data = pd.read_csv('./datas/titanic.csv')
data.dropna(subset=['Age'], inplace=True)

data_1 = data.copy()
data_2 = data.copy()


# 二值化
from sklearn.preprocessing import Binarizer

Age = Binarizer(threshold=17.9).fit_transform(data_1['Age'].values.reshape(-1, 1))
print(np.unique(Age.flatten()))

data_1['Age'] = Age
print(data_1.head())


# 分箱

from sklearn.preprocessing import KBinsDiscretizer

# ordinal-编码为整数，uniform-等宽分箱
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
Age = est.fit_transform(data_2['Age'].values.reshape(-1,1))
print(np.unique(Age.flatten()))

# onehot为分箱后为哑变量
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
Age = est.fit_transform(data_2['Age'].values.reshape(-1,1))
print(Age.toarray())







