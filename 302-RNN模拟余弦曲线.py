
# Pytorch RNN根据正弦sin模拟余弦cos曲线

# 循环神经网络（RNN）让神经网络有了记忆，能够更好的模拟序列化的数据。虽然RNN的原理很简单，但代码特别是参数上，需要花一些时间去理解。
# 以下我们用Pytorch中的RNN类，实现用sin曲线预测cos曲线的模型。

# 1、生成数据集
# 仅演示效果，和模型逻辑无关

# import numpy as np
# import matplotlib.pyplot as plt

# steps = np.linspace(0, 2*np.pi, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)

# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.legend()
# plt.show()



# 2、定义模型

import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state):
        outs, h_state = self.rnn(x, h_state)
        print(outs.shape)
        exit()
        # 用10个输出，拟合10个目标点
        return self.fc(outs), h_state
        

# 2、参数定义和模型实例化
import numpy as np
import torch

input_size = 1
output_size = 1
time_step = 10
hidden_size = 32

net = Net(input_size, hidden_size, output_size)


# 3、模型训练

epoch = 200

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

plt.figure(figsize=(20, 4))
plt.ion()

h_state = None

for i in range(epoch):
    steps = np.linspace(i * np.pi, (i + 1) * np.pi, time_step, dtype=np.float32)

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    y_pred, h_state = net(x, h_state)
    h_state = h_state.data

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i > 150:
        # 模型可视化
        plt.plot(steps, y_np, 'b-')
        plt.plot(steps, y_pred.data.numpy().flatten(), 'r-')
        plt.draw()
        plt.pause(0.05)

plt.ioff()
plt.show()


