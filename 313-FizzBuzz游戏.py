# PyTorch 两层神经网络实现FizzBuzz游戏
# FizzBuzz是一个简单的小游戏，游戏规则如下：
# 1）从1开始往上数数，当遇到3的倍数的时候，说fizz；
# 2）当遇到5的倍数，说buzz；
# 3）当遇到15的倍数，就说fizzbuzz；
# 4）其他情况下则正常数数。
# 现在要求使用神经网络实现FizzBuzz问题。
# FizzBuzz问题本质上是一个四分类问题，即输入一个数字，我们需要将其分为数字本身、Fizz、Buzz、FizzBuzz其中的一类。我们可以搭建一个神经网络，其输入层、隐层、输出层均为全连接层，借助它完成分类任务，进而解决问题。

# 代码示例

# 1、定义转码解码函数

# 可以理解为转化为type
def fb_encode(i):
    if i % 15 == 0:
        return 0
    elif i % 5 == 0:
        return 1
    elif i % 3 == 0:
        return 2
    else:
        return 3

# 根据类型值转化
def fb_decode(i, type):
    return ['fizzbuzz', 'buzz', 'fizz', str(i)][type]

# # 测试
# for i in range(1, 101):
#     print(fb_decode(i, fb_encode(i)))

# 2、定义训练集
# 有了前面的转码解码函数，现在问题就转化成了4分类问题，只需要通过深度学习，找到数值和type的对应关系即可。

# 整数值转化为10位二进制列表
def to_binary(i):
    bstr = str(bin(i))[2:].zfill(10)
    return list(map(int, bstr))

import torch

# 1-100做测试数据，从101开始
x_train = torch.FloatTensor([to_binary(i) for i in range(101, pow(2, 10))])
y_train = torch.LongTensor([fb_encode(i) for i in range(101, pow(2, 10))]) # 分类问题，注意long类型


# 3、定义模型

import torch.nn as nn

D_h = 100
model = nn.Sequential(
    nn.Linear(10, D_h),
    nn.ReLU(D_h),
    nn.Linear(D_h, 4)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 4、模型训练

for it in range(3000):
    for batch_start in range(0, len(x_train), 64):
        batch_end = batch_start + 64
        x = x_train[batch_start:batch_end]
        y = y_train[batch_start:batch_end]

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        print(it, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

# 5、基于模型做预测

import numpy as np

x_test = torch.FloatTensor([to_binary(i) for i in range(1, 101)])
y_pred = model(x_test).max(1)[1].tolist()

# 展示真实值和预测值
for i,type in zip(range(1,101), y_pred):
    print(i, fb_decode(i, fb_encode(i)), fb_decode(i, type))
    
# 计算准确率
np_true = np.array([fb_encode(i) for i in range(1,101)])
np_pred = np.array(y_pred)

print('ACC:', (np_true == np_pred).sum()/len(np_true))

