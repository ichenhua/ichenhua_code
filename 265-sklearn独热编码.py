import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('./datas/titanic.csv')
data.dropna(subset=['Embarked'], inplace=True)
data.reset_index(drop=True, inplace=True)

# 获取编码信息
enc = OneHotEncoder().fit(data[['Sex', 'Embarked']])
print(enc.categories_)
print(enc.get_feature_names_out(['Sex', 'Embarked']))

# 编码
enc_arr = OneHotEncoder().fit_transform(data[['Sex', 'Embarked']]).toarray()
print(enc_arr.shape)

# 编码后数据处理（合并可能会出现多出两条数据的bug，需要先data.reset_index）
columns = enc.get_feature_names_out(['Sex', 'Embarked'])
enc_df = pd.DataFrame(enc_arr, columns=columns)
new_data = pd.concat([data, enc_df], axis=1)
# 删除数据列
new_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
print(new_data.columns)

