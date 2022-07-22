# Pytorch深度学习实现分类算法（Classification）

# 生成随机数据集
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

x, y = make_blobs(200, 2, centers=[[2, 3], [6, 8]])
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

# 建立神经网络
import torch
import torch.nn as nn

class CModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


cmodule = CModule()
# print(cmodule)


# 模型训练
x, y = make_blobs(200, 2, centers=4)
x = torch.FloatTensor(x)
y = torch.tensor(y)

for i in range(10):
    x = torch.tensor(x).type(torch.FloatTensor)
    p_y = cmodule(x)
    print(p_y)
    exit()

# # 优化器
optimizer = torch.optim.Adam(cmodule.parameters(), lr=0.02)
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# for i in range(1000):
#     x = torch.tensor(x).type(torch.FloatTensor)
#     p_y = cmodule(x)
#     y = torch.tensor(y).long()
#     loss = loss_fn(p_y, y)

# 可视化预测结果
ax = plt.axes()
plt.ion()
plt.show()

for i in range(1000):
    p_y = cmodule(x)
    loss = loss_fn(p_y, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 准确率
    p_y = torch.argmax(p_y, 1).data.numpy()
    accuracy = (p_y == y.data.numpy()).sum() / y.size()[0]

    plt.cla()
    plt.scatter(x[:, 0], x[:, 1], c=p_y)
    plt.text(0.75, 0.05 , 'accuracy:' + str(accuracy), transform=ax.transAxes)
    plt.pause(0.3)
