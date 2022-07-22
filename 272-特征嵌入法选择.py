# 嵌入法基本用法
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./datas/digit_recognizor_simple.csv')

x = data.iloc[:, 1:]
y = data.iloc[:, 0]

RFC_ = RFC(random_state=42)

x_embedded = SelectFromModel(RFC_, threshold=0.0005).fit_transform(x, y)
print(x_embedded.shape)  #(1000, 351)

score = cross_val_score(RFC_, x_embedded, y, cv=10).mean()
print(score)  # 0.88


# 学习曲线调参
import numpy as np
import matplotlib.pyplot as plt

scores = []
thresholds = np.linspace(0, RFC_.fit(x, y).feature_importances_.max(), 20)
for ts in thresholds:
    x_embedded = SelectFromModel(RFC_, threshold=ts).fit_transform(x, y)
    score = cross_val_score(RFC_, x_embedded, y, cv=10).mean()
    scores.append(score)

plt.plot(thresholds, scores)
plt.xticks(thresholds)
plt.show()




