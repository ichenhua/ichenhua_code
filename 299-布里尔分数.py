# 1、导入并拆分数据集

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ds = load_breast_cancer()
x, y = ds.data, ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 2、训练并计算概率

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 高斯贝叶斯
gnb = GaussianNB()
gnb.fit(x_train, y_train)
prob_gnb = gnb.predict_proba(x_test)

# 逻辑回归
lr = LogisticRegression()
lr.fit(x_train, y_train)
prob_lr = lr.predict_proba(x_test)

# SVM，获取点到决策边界的距离，距离越远可信度越高，归一化后当近似概率值
svc = SVC(probability=True)
svc.fit(x_train, y_train)
prob_svc = svc.predict_proba(x_test)

# 3、计算布里尔分数

from sklearn.metrics import brier_score_loss

score_gnb = brier_score_loss(y_test, prob_gnb[:, 1], pos_label=1)
print('score gnb:', score_gnb)
# score gnb: 0.075

score_lr = brier_score_loss(y_test, prob_lr[:, 1], pos_label=1)
print('score lr:', score_lr)
# score lr: 0.031

score_svc = brier_score_loss(y_test, prob_svc[:, 1], pos_label=1)
print('score svc:', score_svc)
# score svc: 0.044

# 从分数值大小判断：逻辑回归效果最好，贝叶斯次之，SVM效果最差。

# 3、计算对数损失

from sklearn.metrics import log_loss

print('loss gnb:', log_loss(y_test, prob_gnb))
print('loss lr:', log_loss(y_test, prob_lr))
print('loss svc:', log_loss(y_test, prob_svc))

# loss gnb: 0.7082332572488389
# loss lr: 0.10596180857071849
# loss svc: 0.16000605663068623

# 从损失值的大小判断：逻辑回归效果最好，SVM次之，贝叶斯效果最差，这和前文介绍布里尔分数顺序不太一致。
