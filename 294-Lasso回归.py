
# 1、导入并拆分数据集

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

ds = load_boston()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# 2、建模并查看系数

from sklearn.linear_model import LinearRegression, Ridge, Lasso

linear = LinearRegression()
linear.fit(x_train, y_train)
print(linear.coef_)
# [-1.21310401e-01  4.44664254e-02  1.13416945e-02  2.51124642e+00
#  -1.62312529e+01  3.85906801e+00 -9.98516565e-03 -1.50026956e+00
#   2.42143466e-01 -1.10716124e-02 -1.01775264e+00  6.81446545e-03
#  -4.86738066e-01]

ridge = Ridge()
ridge.fit(x_train, y_train)
print(ridge.coef_)
#[-1.18308575e-01  4.61259764e-02 -2.08626416e-02  2.45868617e+00
# -8.25958494e+00  3.89748516e+00 -1.79140171e-02 -1.39737175e+00
#  2.18432298e-01 -1.16338128e-02 -9.31711410e-01  7.26996266e-03
# -4.94046539e-01]

lasso = Lasso()
lasso.fit(x_train, y_train)
print(lasso.coef_)
# [-0.06586193  0.04832933 -0.          0.         -0.          0.86898466
#  0.01217999 -0.75109378  0.2000743  -0.01395062 -0.84602363  0.00668818
# -0.73266568]

# 观察对比发现，Lasso中，有3项特征系数为0，如果我们是做特征选择，可以考虑将这三项特征删除。




