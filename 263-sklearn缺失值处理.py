# Sklearn缺失值处理-填充和删除

# 在机器学习和数据挖掘的实际场景中，数据一般都不会像Sklearn给我们提供的数据那么完美，难免会有一些缺失值，所以在做数据预处理时，对缺失值的处理，是必不可少的一个步骤。
# 本文介绍两种常见的处理方式：填充和删除，缺失比例较大的字段一般采用填充，缺失比例很小的数据，一般会删除整行。本文依然以泰坦尼克号的数据做演示。

# Pandas处理数据
# 一般场景下，用 Pandas 处理数据会更简单。
import pandas as pd
data = pd.read_csv('./datas/titanic.csv')
# data.info()
# 5   Age          714 non-null    float64
# 11  Embarked     889 non-null    object 

# Age列有177个缺失值，填充中位数
data['Age'] = data['Age'].fillna(data['Age'].median())

# Embarked列有2个缺失值，直接删除对应列
# 删除DataFrame里某一列有空值的行
data.dropna(subset=['Embarked'], inplace=True)


# Sklearn处理数据
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv('./datas/titanic.csv')

# 填充Age字段
Age = data['Age'].values.reshape(-1, 1)  # sklearn特征矩阵必须是二维

imp_mean = SimpleImputer()  # 默认填充均值
imp_median = SimpleImputer(strategy='median')  # 填充中值
imp_0 = SimpleImputer(strategy='constant', fill_value=0)  # 填充0

Age_mean = imp_mean.fit_transform(Age)
Age_median = imp_median.fit_transform(Age)
Age_0 = imp_0.fit_transform(Age)

data['Age'] = Age_median

# 填充Embarked字段
Embarked = data['Embarked'].values.reshape(-1, 1)
imp_mode = SimpleImputer(strategy='most_frequent') # 众数填充
data['Embarked'] = imp_mode.fit_transform(Embarked)

data.info()


