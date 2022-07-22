# 澳大利亚天气预测项目特征工程

# 澳大利亚天气预测，是Kaggle上一个非常接近真实场景的数据集，
# 因为其数据结构复杂，前期需要做大量的数据预处理，所以本文先介绍澳大利亚天气数据集的特征工程部分，下节课再进行建模分析。

# Kaggle下载地址：https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

# Date：观察特征的那一天
# Location：观察的城市
# MinTemp：当天最低温度（摄氏度）
# MaxTemp：当天最高温度（摄氏度）温度都是 string
# Rainfall：当天的降雨量（单位是毫米mm）
# Evaporation：一个凹地上面水的蒸发量（单位是毫米mm），24小时内到早上9点
# Sunshine：一天中出太阳的小时数
# WindGustDir：最强劲的那股风的风向，24小时内到午夜
# WindGustSpeed：最强劲的那股风的风速（km/h），24小时内到午夜
# WindDir9am：上午9点的风向
# WindDir3pm：下午3点的风向
# WindSpeed9am：上午9点之前的十分钟里的平均风速，即 8:50~9:00的平均风速，单位是（km/hr）
# WindSpeed3pm：下午3点之前的十分钟里的平均风速，即 14:50~15:00的平均风速，单位是（km/hr）
# Humidity9am：上午9点的湿度
# Humidity3pm：下午3点的湿度
# Pressure9am：上午9点的大气压强（hpa）
# Pressure3pm：下午3点的大气压强
# Cloud9am：上午9点天空中云的密度，取值是[0, 8]，以1位一个单位，0的话表示天空中几乎没云，8的话表示天空中几乎被云覆盖了
# Cloud3pm：下午3点天空中云的密度
# Temp9am：上午9点的温度（单位是摄氏度）
# Temp3pm：下午3点的温度（单位是摄氏度）
# RainTodayBoolean: 今天是否下雨
# RainTomorrow：明天是否下雨（标签值）

# 1、导入数据集和初步探索
# 探索发现，特征数据有不同程度的缺失

import pandas as pd

weather = pd.read_csv('./datas/weatherAUS5000.csv', index_col=0)
# weather.info()
# print(weather.isnull().mean())

# 2、切分数据集

from sklearn.model_selection import train_test_split

x = weather.iloc[:, :-1]
y = weather.iloc[:, -1]

# print(y.unique())  #['Yes' 'No']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# print(y_train.value_counts()) # No 2722 Yes 778
# print(y_test.value_counts()) # No 1133 Yes 367
# 打印发现，训练集和测试集都存在样本不均衡问题

# 3、标签编码

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder().fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

# print(y_train.shape)

# 4、处理日期特征
# 天气预测可能和日期中的月份有关系，所以提取日期中的月份

x_train['Month'] = x_train['Date'].str.split('-', expand=True)[1].astype('int')
x_train.drop('Date', axis=1, inplace=True)

x_test['Month'] = x_test['Date'].str.split('-', expand=True)[1].astype('int')
x_test.drop('Date', axis=1, inplace=True)

# print(len(x_train['Month'].unique()))

# 5、处理地点特征
# 观测点所在城市，对应所在气候区域划分，可能对天气预测有影响，所以将城市替换成气候区域。
# 本过程比较复杂，参见上一篇文章：澳大利亚观测点城市气候区域划分 http://www.ichenhua.cn/read/289

city_climate = pd.read_csv('./datas/sample_city_climate.csv', index_col=0)
# 用气候区域替换观测点城市，并去掉空格
climate_dict = city_climate['Climate'].to_dict()
x_train['Climate'] = x_train['Location'].apply(lambda x: climate_dict[x].strip())
x_train.drop('Location', axis=1, inplace=True)
x_test['Climate'] = x_test['Location'].apply(lambda x: climate_dict[x].strip())
x_test.drop('Location', axis=1, inplace=True)

# 6、众数填补分类缺失值

cate_col = x_train.columns[x_train.dtypes == 'object'].tolist()
# Cloud9am、Cloud3pm、Month虽然是数字，但风力等级，应该当分类处理
cate_col += ['Cloud9am', 'Cloud3pm', 'Month']

from sklearn.impute import SimpleImputer

impmost = SimpleImputer(strategy='most_frequent')
impmost.fit(x_train.loc[:, cate_col])

x_train.loc[:, cate_col] = impmost.transform(x_train.loc[:, cate_col])
x_test.loc[:, cate_col] = impmost.transform(x_test.loc[:, cate_col])

# x_test.info()

# 7、分类特征编码
# 至此数据全部变为了数值型
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
encoder.fit(x_train.loc[:, cate_col])

x_train.loc[:, cate_col] = encoder.transform(x_train.loc[:, cate_col])
x_test.loc[:, cate_col] = encoder.transform(x_test.loc[:, cate_col])

# x_train.info()

# 8、均值填补连续数据缺失值

cols = x_train.columns.tolist()
seri_col = list(set(cols) - set(cate_col))

impmean = SimpleImputer()
impmean.fit(x_train.loc[:, seri_col])

x_train.loc[:, seri_col] = impmean.transform(x_train.loc[:, seri_col])
x_test.loc[:, seri_col] = impmean.transform(x_test.loc[:, seri_col])

# print(x_test.isnull().mean())

# 9、连续数据标准化

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train.loc[:, seri_col])

x_train.loc[:, seri_col] = scaler.transform(x_train.loc[:, seri_col])
x_test.loc[:, seri_col] = scaler.transform(x_test.loc[:, seri_col])

# print(x_test.describe().T)


# SVM建模预测澳大利亚天气
# 前面两篇文章介绍了澳大利亚天气数据集的特征工程，将数据处理到了可以建模的程度，本文介绍SVM建模来做天气预测。
# 同时在线性模型基础上，介绍准确率和召回率的平衡调节方法。

# 10、建模和模型评估

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in kernels:
    clf = SVC(kernel=k, degree=1, gamma='auto').fit(x_train, y_train)
    score = clf.score(x_test, y_test)  #准确率
    # 其他指标
    y_pred = clf.predict(x_test)
    recall = recall_score(y_test, y_pred)  #召回率
    auc = roc_auc_score(y_test, clf.decision_function(x_test))
    print('%s score:%.3f, recall:%.3f, auc:%.3f' % (k, score, recall, auc))
    # linear score:0.831, recall:0.431, auc:0.862
    # poly score:0.834, recall:0.422, auc:0.862
    # rbf score:0.806, recall:0.308, auc:0.826
    # sigmoid score:0.637, recall:0.172, auc:0.437

# 以上训练结果中，准确率还勉强可以，但recall都不高，而且在特征工程阶段，我们就发现数据存在不均衡的问题。
# 所以下面我们还需要在召回率，准确率，以及两者的平衡方向，继续通过调参来优化模型。

# 11、追求更高的recall
# 要追求更高的recall，就要不惜一切代价判断出少数类，以下我们用class_weight=balanced和固定值，来调节权重。

clf = SVC(kernel=k, degree=1, class_weight='balanced', gamma='auto').fit(x_train, y_train)
# linear score:0.786, recall:0.749, auc:0.861
# poly score:0.794, recall:0.755, auc:0.862
# rbf score:0.793, recall:0.597, auc:0.825
# sigmoid score:0.501, recall:0.376, auc:0.437

# class_weight={1:10} 表示1类别占10份，隐藏了0:1
clf = SVC(kernel=k, degree=1, class_weight={1: 10}, gamma='auto').fit(x_train, y_train)
# linear score:0.636, recall:0.905, auc:0.857
# poly score:0.631, recall:0.910, auc:0.857
# rbf score:0.783, recall:0.575, auc:0.812
# sigmoid score:0.245, recall:1.000, auc:0.437

# 随着class_weight的上升，在线性模型上得到了很高的recall值，但这样调整显然无法得到一个临界值。



# 12、追求平衡点
# 经过前面的尝试，我们发现当前数据集，线性模型的效果最好，所以我们在线性模型的基础上，来画ROC曲线，获得临界阈值。

clf = SVC(kernel='linear', degree=1, class_weight='balanced', gamma='auto').fit(x_train, y_train)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(x_test))
area = roc_auc_score(y_test, clf.decision_function(x_test))

# 画ROC曲线
# plt.plot(fpr, tpr, label='ROC curve (area=%.3f)' % area)
# plt.plot([0, 1], [0, 1], c='k', linestyle='--')
# plt.legend()
# plt.show()

# 找到平衡点对应threshold
idx = np.argmax(tpr - fpr)
threshold = thresholds[idx]
# 设定阈值下的预测值
prob = clf.decision_function(x_test)
y_pred_t = (prob >= threshold).astype('int')

# 计算平衡点的准确率和召回率
score = accuracy_score(y_test, y_pred_t)
recall = recall_score(y_test, y_pred_t)

print('score:%.3f, recall:%.3f' % (score, recall))
# score:0.801, recall:0.728

# 至此，这份数据集经过特征处理后，在SVM模型下的效果已经基本到极限了。
# 如果还想得到更好的效果，可以继续尝试其他模型，但亲测决策树、随机森林、KNN都未能超过SVM的效果。
# 另一个优化的方向，就是特征工程，但可能要有一定的专业知识，大家有时间还可以继续探索。









