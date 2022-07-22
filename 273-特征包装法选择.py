
# 包装法基本用法
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./datas/digit_recognizor_simple.csv')

x = data.iloc[:, 1:]
y = data.iloc[:, 0]

RFC_ = RFC(random_state=42)
x_wrapper = RFE(RFC_, n_features_to_select=300, step=50).fit_transform(x,y)

score = cross_val_score(RFC_, x_wrapper, y, cv=10).mean()
print(score)  # 0.872


# 学习曲线调参
import matplotlib.pyplot as plt

scores = []
for i in range(1, x.shape[1], 50):
    x_wrapper = RFE(RFC_, n_features_to_select=i, step=50).fit_transform(x, y)
    score = cross_val_score(RFC_, x, y, cv=10).mean()
    scores.append(score)

plt.plot(range(1, x.shape[1], 50), scores)
plt.xticks(range(1, x.shape[1], 50))
plt.show()


