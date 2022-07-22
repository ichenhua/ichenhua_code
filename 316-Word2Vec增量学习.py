# Word2Vec迭代加载文本和模型增量学习

# 前文介绍了Word2Vec的使用流程，需要先导入文本，再训练模型。但真实场景中，可能会面临两个问题，
# 一是训练数据不是一个文件，而是很多个小文档；二是模型也不是一成不变的，可能会有更新的需求。
# 下面就来解决这两个问题，迭代加载文本和模型增量学习。

# 代码示例

# 1、迭代加载文本
# 参考文档：https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

from glob import glob
import jieba
from gensim.models import Word2Vec

# 定义文本加载类
def iter_documents(top_dir):
    for file_path in glob(top_dir + '*.txt'):
        yield open(file_path).read()

class TxtSubdirsCorpus(object):
    def __init__(self, top_dir, stopwords):
        self.top_dir = top_dir
        self.stopwords = stopwords

    def __iter__(self):
        for text in iter_documents(self.top_dir):
            yield self.process(text)

    def process(self, text):
        # 分词并过滤停用词
        words = jieba.lcut(text)
        return [word for word in words if word not in self.stopwords and len(word) >= 2]

# 加载停用词表
stopwords_str = open('./datas/stopwords.txt').read()
stopwords = stopwords_str.split('\n')

# jieba添加自定义词语，防止误拆

name_list = [
    '沙瑞金', '田国富', '高育良', '侯亮平', '钟小艾', '陈岩石', '欧阳菁', '易学习', '王大路', '蔡成功', '孙连城', '季昌明', '丁义珍', '郑西坡', '赵东来', '高小琴', '赵瑞龙',
    '林华华', '陆亦可', '刘新建', '刘庆祝', '赵德汉'
]
for name in name_list:
    jieba.add_word(name)

corpus = TxtSubdirsCorpus('./datas/in_the_name_of_people/part1/', stopwords)

model = Word2Vec(corpus, min_count=1, vector_size=20)
print(len(model.wv.index_to_key))  # 2591


# 2、模型增量学习

corpus_new = TxtSubdirsCorpus('./datas/in_the_name_of_people/part2/', stopwords)
# 添加新的章节
model.build_vocab(corpus_new, update=True)
# 进行训练
model.train(corpus_new, total_examples=model.corpus_count, epochs=model.epochs)
print(len(model.wv.index_to_key))  # 3506



