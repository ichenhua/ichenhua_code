
# Pytorch LSTM模型生成藏头古诗

# 循环神经网络，被广泛应用在自然语言处理领域(NLP)，本文就使用RNN的一个改进模型LSTM来做一个小案例，生成藏头古诗。

# 1、训练词向量

from gensim.models import Word2Vec

class Corpus():
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                # 将句子拆分成字
                yield ' '.join(line).split(' ')

corpus = Corpus('./datas/poetry_7.txt')
model = Word2Vec(corpus, min_count=1, vector_size=100)

# 保存词向量相关数据
vectors = model.wv.vectors
key_to_index = model.wv.key_to_index
index_to_key = model.wv.index_to_key

# 2、构建数据集

import torch.utils.data as data
import numpy as np

class PoetryDataset(data.Dataset):
    def __init__(self, file_path, vectors, key_to_index):
        super().__init__()
        self.vectors = vectors
        self.key_to_index = key_to_index
        # 读取文件
        with open(file_path) as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip()
        x_idx = []
        y_idx = []
        # 错位，建立前后关系
        for xs, ys in zip(line[:-1], line[1:]):
            x_idx.append(self.key_to_index[xs])
            y_idx.append(self.key_to_index[ys])
        x = vectors[x_idx]
        return x, np.array(y_idx)

dataset = PoetryDataset('./datas/poetry_7.txt', vectors, key_to_index)

loader = data.DataLoader(dataset, shuffle=True, batch_size=100)

# 3、定义模型

import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        # 2层，双向LSTM
        self.lstm = nn.LSTM(param['D_in'], param['D_hidden'], num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0, 1)
        # 双向，输出隐层x2
        self.linear = nn.Linear(2 * param['D_hidden'], param['D_out'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=param['lr'])

    def forward(self, x, h_n, c_n):
        x, (h_n, c_n) = self.lstm(x)
        x = self.dropout(x)
        # 展平后才能输入fc层
        x = self.flatten(x)
        out = self.linear(x)
        return out, (h_n, c_n)


# 4、模型训练
# 计算量不是一般的大，GPU自然少不了。

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device', device)

param = {
    'D_in': 100,
    'D_hidden': 128,
    'D_out': len(index_to_key),
    'lr': 1e-4,
}

net = Net(param).to(device)

# 定义初始参数
h_n = c_n = None
for e in range(1000):
    for i, (x, y) in enumerate(loader):
        # 数据迁移到GPU
        x = x.to(device)
        y = y.to(device)
        # 训练模型
        y_pred, (h_n, c_n) = net(x, h_n, c_n)
        # 要注意和输出的维度保持一致
        loss = net.loss_fn(y_pred, y.view(-1))

        net.optimizer.zero_grad()
        loss.backward()
        net.optimizer.step()
        
        if e % 50 ==0 and e % 50 == 0:
            print(e, i, loss)
            torch.save(net, f'./net_{e}.m')
    torch.save(net, './net.m')

# 5、古诗生成

# 随机生成一首诗
word_idx = np.random.randint(len(key_to_index))
result = index_to_key[word_idx]
# 初始化输入参数
h_g = torch.zeros(4, 100, 128)
c_g = torch.zeros(4, 100, 128)
# 根据第一个字，生成后面的31个字
for i in range(31):
    x_g = torch.tensor(vectors[word_idx][None][None]).to(device)
    out, (h_g, c_g) = net(x_g, h_g, c_g)
    word_idx = torch.argmax(out).item()
    result += index_to_key[word_idx]

print(result)

# 藏头诗

word_list = ['独', '每', '遥', '遍']
points = ['，', '；', '，', '。']

result = ''
for w,p in zip(word_list, points):
    result += w
    # 防止出现生僻字
    try:
        word_idx = key_to_index[w]
    except KeyError:
        word_idx = np.random.randint(len(key_to_index))
    h_g = torch.zeros(4, 100, 128)
    c_g = torch.zeros(4, 100, 128)
    # 生成后面6个字
    for i in range(6):
        x_g = torch.tensor(vectors[word_idx][None][None]).to(device)
        out, (h_g, c_g) = net(x_g, h_g, c_g)
        word_idx = torch.argmax(out).item()
        result += index_to_key[word_idx]
    result += p
    
print(result)

# 很遗憾，不论是完整的，还是藏头的诗，效果都不是很好。原因可能是模型太简单，没有Attention机制，后面再优化。
# 本文的主要目的，还是熟悉LSTM结构，和序列生成的套路，生成句子的场景大多是自己玩玩，实际生产环境中用到的可能性不大。
