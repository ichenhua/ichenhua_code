import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./datas/titanic.csv')
data.dropna(subset=['Embarked'], inplace=True)

le = LabelEncoder()
le.fit(data['Embarked'])  # 生成classes_
res = le.transform(['S', 'Q', 'C'])  # 编码

print(res)
print(le.classes_)  # 查看编码list

# 逆转
res = le.inverse_transform([0, 1, 2])
print(res)

# 不需要展示中间过程时简写
from sklearn.preprocessing import LabelEncoder
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])





from sklearn.preprocessing import OrdinalEncoder
data = pd.read_csv('./datas/titanic.csv')
data.dropna(subset=['Embarked'], inplace=True)

# 注意：本方法只接受二维特征
cates = OrdinalEncoder().fit(data[['Sex', 'Embarked']]).categories_
print(cates)

data[['Sex', 'Embarked']] = OrdinalEncoder().fit_transform(data[['Sex', 'Embarked']])
print(data['Sex'].unique())

