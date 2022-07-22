# 1、模型训练

from gensim.models import Word2Vec

sentences = [
    ['my', 'cat', 'sat', 'on', 'my', 'bed'],
    ['my', 'dog', 'sat', 'on', 'my', 'knees'],
    ['my', 'bird', 'was', 'shut', 'in', 'a', 'cage'],
]

model = Word2Vec(sentences, min_count=1, vector_size=2)
# 重要参数：
# sentences: list或者可迭代的对象
# vector_size: 词向量维度，默认100
# window: 窗口大小，即词向量上下文最大距离，默认5
# min_count: 需要计算词向量的最小词频，默认5，小语料需要调整


# 2、模型保存和加载

# # 保存模型
# model.save('./w2v.m')
# # 加载模型
# Word2Vec.load('./w2v.m')

# model.wv.save('wv.m')

# 3、重要属性

# 词向量矩阵
print(model.wv.vectors)

# 查看所有词汇
print(model.wv.index_to_key)

# 查看词汇对应索引
print(model.wv.key_to_index)

# 查看所有词出现的次数
for word in model.wv.index_to_key:
    print(word, model.wv.get_vecattr(word, 'count'))

# 4、常用方法

# 根据词查词向量
print(model.wv.get_vector('cat'))  # word or index
print(model.wv.get_vector(12))

# 查看某个词相近的词
print(model.wv.similar_by_word('cat'))  #
print(model.wv.similar_by_key(12))
# 根据向量查询相近的词
vec = model.wv.get_vector(12)
print(model.wv.similar_by_key(vec))

# 根据给定的条件推断相似词
print(model.wv.most_similar(positive=['cat', 'dog'], negative=['bird']))

# 查看两个词相似度
print(model.wv.similarity('cat', 'dog'))

# 给定上下文词汇作为输入，可以获得中心词汇的概率分布
print(model.predict_output_word(['cat', 'bed'], topn=10))

# 寻找离群词
print(model.wv.doesnt_match(['cat', 'dog', 'bed', 'man']))
