# 1、卡方过滤
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC

data = pd.read_csv('./datas/digit_recognizor_simple.csv')
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

x_chi = SelectKBest(chi2, k=300).fit_transform(x, y)

score = cross_val_score(RFC(random_state=42), x_chi, y, cv=5).mean()
print(score)  # 0.855

#互信息法
from sklearn.feature_selection import mutual_info_classif

h = mutual_info_classif(x, y)
k = h.shape[0] - (h<=0).sum()
print(k)  #615



# 检验F
from sklearn.feature_selection import f_classif

F, p = f_classif(x, y)
k = F.shape[0] - (p >= 0.05).sum()
print(k)  #643


exit()

# k值调参
import matplotlib.pyplot as plt

score_l = []
for i in range(200, 401, 10):
    x_chi = SelectKBest(chi2, k=i).fit_transform(x, y)

    score = cross_val_score(RFC(random_state=42), x_chi, y, cv=5).mean()
    score_l.append(score)

plt.plot(range(200, 401, 10), score_l)
plt.show()

# P值调参
chi, p = chi2(x, y)
# k的取值，可以用总特征数，减去p大于设置值的总数
k = chi.shape[0] - (p > 0.01).sum()
print(k)  # 784

# 很遗憾，k值最后结果和特征数相等，说明卡方过滤对该组数据不适用，可以考虑只用前面学过的特征方差过滤。
