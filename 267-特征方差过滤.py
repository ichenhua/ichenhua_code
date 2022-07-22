# 导入数据
import pandas as pd

data = pd.read_csv('./datas/digit_recognizor_simple.csv')
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(x.shape, y.shape)  # (1000, 784) (1000,)


# 方差过滤
from sklearn.feature_selection import VarianceThreshold
# 默认消除方差为0的特征
x_var0 = VarianceThreshold().fit_transform(x)
print(x_var0.shape)  # (1000, 612)

# 方差中位数过滤，可以消掉大约一半的特征
import numpy as np

x_varmd = VarianceThreshold(np.median(x.var().values)).fit_transform(x)
print(x_varmd.shape)  # (1000, 391)

# 当特征为二分类时，特征的取值就是伯努利随机变量，方差var[x] = p(1-p)
# 假设p=0.8，即二分类特征中某种分类占到80%以上的时候删除特征
x_varb = VarianceThreshold(0.8 * (1 - 0.8)).fit_transform(x)
print(x_varb.shape)


# KNN算法
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score

score = cross_val_score(KNN(), x, y, cv=5).mean()
print(score)  # 0.837  2.522 seconds

score = cross_val_score(KNN(), x_varmd, y, cv=5).mean()
print(score)  # 0.839  1.504 seconds


# 随机森林
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

score = cross_val_score(RFC(), x, y, cv=5).mean()
print(score)  # 0.880  3.795 seconds

score = cross_val_score(RFC(n_estimators=10), x_varmd, y, cv=5).mean()
print(score)  # 0.798  1.843 seconds





