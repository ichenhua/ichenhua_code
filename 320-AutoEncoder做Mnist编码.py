# 自编码器AutoEncoder做Mnist编码

# 在深度学习中，自编码器(AutoEncoder, AE)是一种无监督的神经网络模型，它可以学习到输入数据的隐含特征，这称为编码(coding)，同时用学习到的新特征可以重构出原始输入数据，称之为解码(decoding)。从直观上来看，自动编码器可以用于特征降维，类似主成分分析PCA，但是其相比PCA其性能更强，这是由于神经网络模型可以提取更有效的新特征。
# 本文用一个简单的Pytorch自带数据集Mnist，来演示编码和解码过程。

#  1、导入模块
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt


# 2、定义模型
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_shape = x.size()
        # flatten
        x = x.view(x_shape[0], -1)
        # encode
        x = self.encoder(x)
        # decode
        x = self.decoder(x)
        # reshape
        x = x.view(x_shape)
        return x


if __name__ == '__main__':
    # 3、加载数据集
    train_data = datasets.MNIST(
        './datas/mnist',
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True,
    )
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # 4、模型实例化和训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 5、训练效果可视化
    show_img_num = 5  # 可视化展示图片数量
    fig, axes = plt.subplots(2, 5, figsize=(5, 2))
    plt.ion()

    for e in range(100):
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat = net(x)
            loss = criterion(x_hat.to(device), x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('>>', e, i, loss.item())
                for n in range(show_img_num):
                    # 展示原图
                    axes[0][n].clear()
                    axes[0][n].imshow(x[n].squeeze(0).data.numpy())
                    axes[0][n].axis('off')
                    # 展示生成图
                    axes[1][n].clear()
                    axes[1][n].imshow(x_hat[n].squeeze(0).data.numpy())
                    axes[1][n].axis('off')
                    # 画图
                    plt.draw()
                    plt.pause(0.1)

    plt.show()
    plt.ioff()
