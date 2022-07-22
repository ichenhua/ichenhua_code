# 1、导入数据集
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

ds = load_boston()
x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



# 2、岭回归建模
from sklearn.linear_model import Ridge

reg = Ridge(random_state=0)
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test)) #0.666

# 相比之前直接用线性回归求解r2值，岭回归反而降低了，这说明这个波斯顿房价数据集并没有共线性问题，岭回归并不能提升模型表现。



# 3、学习曲线调参
# 当然岭回归中alpha参数时可以调整的，所以我们还是用学习曲线来尝试调整参数，看模型表现是否能提升。

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

score_r = []
score_l = []
for i in range(1, 300):
    ridge = Ridge(alpha=i, random_state=1)
    score = cross_val_score(ridge, x, y, cv=5).mean() 
    score_r.append(score)
    # 线性模型对比（不会有变化，放到循环外亦可）
    linear = LinearRegression()
    score = cross_val_score(linear, x, y, cv=5).mean()
    score_l.append(score)

plt.plot(range(1,300), score_r, label='ridge model')
plt.plot(range(1,300), score_l, label='linear model')
plt.legend()
plt.show()
