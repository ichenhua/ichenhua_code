# GraphViz在线绘制Sklearn红酒数据集决策树

# Sklearn红酒数据集，是一份非常适合用来做决策树模型数据集，本文介绍使用GraphViz在线工具，来绘制一个Sklearn红酒数据集决策树。

# 1、导入并拆分数据集
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# 2、训练模型
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42, criterion="gini")
clf.fit(x_train, y_train)

score_train = clf.score(x_train, y_train)
print(score_train)

score_test = clf.score(x_test, y_test)
print(score_test)

# 3、导出模型
from sklearn.tree import export_graphviz

feature_names = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
dot_data = export_graphviz(
    clf
    ,out_file='./wine.dot'  # 输出文件
    ,feature_names=feature_names   # 特征名称
    ,class_names=['赤霞珠', '黑皮诺', '梅洛']   # 分类名称
    ,filled=True   # 是否填充颜色
    ,rounded=True   # 是否圆角效果
)


# 4、图格式化
# http://dreampuf.github.io/GraphvizOnline/

# digraph Tree {
# node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
# edge [fontname="helvetica"] ;
# 0 [label="颜色强度 <= 3.825\ngini = 0.664\nsamples = 124\nvalue = [38, 47, 39]\nclass = 黑皮诺", fillcolor="#ecfdf3"] ;
# 1 [label="od280/od315稀释葡萄酒 <= 3.73\ngini = 0.124\nsamples = 45\nvalue = [3, 42, 0]\nclass = 黑皮诺", fillcolor="#47e78a"] ;
# 0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
# 2 [label="灰 <= 3.07\ngini = 0.045\nsamples = 43\nvalue = [1, 42, 0]\nclass = 黑皮诺", fillcolor="#3ee684"] ;
# 1 -> 2 ;
# 3 [label="gini = 0.0\nsamples = 42\nvalue = [0, 42, 0]\nclass = 黑皮诺", fillcolor="#39e581"] ;
# 2 -> 3 ;
# 4 [label="gini = 0.0\nsamples = 1\nvalue = [1, 0, 0]\nclass = 赤霞珠", fillcolor="#e58139"] ;
# 2 -> 4 ;
# 5 [label="gini = 0.0\nsamples = 2\nvalue = [2, 0, 0]\nclass = 赤霞珠", fillcolor="#e58139"] ;
# 1 -> 5 ;
# 6 [label="类黄酮 <= 1.785\ngini = 0.556\nsamples = 79\nvalue = [35, 5, 39]\nclass = 梅洛", fillcolor="#f4edfd"] ;
# 0 -> 6 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
# 7 [label="灰的碱性 <= 17.15\ngini = 0.049\nsamples = 40\nvalue = [0, 1, 39]\nclass = 梅洛", fillcolor="#843ee6"] ;
# 6 -> 7 ;
# 8 [label="gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]\nclass = 黑皮诺", fillcolor="#39e581"] ;
# 7 -> 8 ;
# 9 [label="gini = 0.0\nsamples = 39\nvalue = [0, 0, 39]\nclass = 梅洛", fillcolor="#8139e5"] ;
# 7 -> 9 ;
# 10 [label="脯氨酸 <= 724.5\ngini = 0.184\nsamples = 39\nvalue = [35, 4, 0]\nclass = 赤霞珠", fillcolor="#e88f50"] ;
# 6 -> 10 ;
# 11 [label="gini = 0.0\nsamples = 4\nvalue = [0, 4, 0]\nclass = 黑皮诺", fillcolor="#39e581"] ;
# 10 -> 11 ;
# 12 [label="gini = 0.0\nsamples = 35\nvalue = [35, 0, 0]\nclass = 赤霞珠", fillcolor="#e58139"] ;
# 10 -> 12 ;
# }