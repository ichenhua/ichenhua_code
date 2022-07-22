# Pytorch深度学习实现线性回归（Linear Regression）

# 创建x变量
import torch
x = torch.linspace(-1, 1, 100, requires_grad=True)
x = x.view(-1, 1)  # change shape
y = x**2 + 0.2 * torch.rand(x.size())

# 画图
# from matplotlib import pyplot as plt
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 建立网络模型
# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html#
import torch.nn as nn
class LModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

lmodule = LModule()


from matplotlib import pyplot as plt
plt.ion()   # 画图
plt.show()

# 训练模型
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
optimizer = torch.optim.SGD(lmodule.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for t in range(1000):
    pred_y = lmodule(x)
    loss = loss_fn(pred_y, y)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pred_y.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
    
