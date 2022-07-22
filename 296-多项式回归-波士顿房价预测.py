
# 1、导入数据集，并获取特征名称
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
x = iris.data
y = iris.target
columns = iris.feature_names

print(columns)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']

# 2、多项式回归

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x = poly.fit_transform(x)

# 当数据有标签时，可以查看拓展后的属性组合
print(poly.get_feature_names(columns))

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 
# 'sepal length (cm)^2', 'sepal length (cm) sepal width (cm)', 'sepal length (cm) petal length (cm)', 'sepal length (cm) petal width (cm)', 
# 'sepal width (cm)^2', 'sepal width (cm) petal length (cm)', 'sepal width (cm) petal width (cm)', 
# 'petal length (cm)^2', 'petal length (cm) petal width (cm)', 'petal width (cm)^2']

