import numpy as np
from torch.utils import data
import os, glob
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GestureDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.data = self._read_file(root, train)
        self.transfrom = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data[index]
        img = Image.open(file_path)
        if self.transfrom:
            img = self.transfrom(img)
        return img, label

    # 读取文件
    def _read_file(self, root, train):
        dir_name = 'train' if train else 'test'
        dir_path = os.path.join(root, dir_name)
        lst = []
        if train:
            for dir in self._list_file(dir_path):
                for file_path in self._list_file(dir):
                    # ./datas/train/9/IMG_5805.JPG
                    label = file_path.split('/')[-2]
                    lst.append((file_path, label))
        else:
            for file_path in self._list_file(dir_path):
                # ./datas/test/example_9.JPG
                label = file_path[-5]
                lst.append((file_path, label))
        return lst

    def _list_file(self, dir_path):
        return glob.glob(dir_path + '/*')


# GestureModule(
#   (conv1): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv2): Sequential(
#     (0): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv3): Sequential(
#     (0): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): Dropout2d(p=0.4, inplace=False)
#     (3): ReLU()
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (out): Linear(in_features=36864, out_features=10, bias=True)
# )


# 定义网络模型
class GestureModule(nn.Module):

    def __init__(self, mudule_path):
        self.mudule_path = mudule_path

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),  #(64, 100, 100)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  #(64, 50, 50)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),  #(64, 50, 50)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  #(128, 25, 25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),  #(256, 25, 25)
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.4),
            nn.ReLU(),
            nn.MaxPool2d(2)  #(256, 12, 12)
        )
        self.out = nn.Linear(256 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def save(self):
        torch.save(self, self.mudule_path)

    @staticmethod
    def load(mudule_path):
        if os.path.exists(mudule_path):
            return torch.load(mudule_path)


def train(module_path):
    #加载训练数据
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([100, 100])])
    root = './datas'
    ds = GestureDataset(root, transform=trans)

    # 加载训练数据
    loader = DataLoader(ds, batch_size=50, shuffle=True)

    # 实例化模型
    module = GestureModule(module_path)

    # 训练模型
    optimizer = torch.optim.Adam(module.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        min_loss = 1000
        for x, y in loader:
            p_y = module(x)
            y = np.array(y, dtype='float')
            # 需要注意长度问题
            y = F.one_hot(torch.from_numpy(y).long(), 10)

            loss = loss_fn(p_y.float(), y.float())

            print('epoch:', epoch, ' loss:', loss.item())
            # 判断loss，更新模型文件
            if float(loss.item()) < min_loss:
                module.save()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    module_path = './module.pkl'
    if not os.path.exists(module_path):
        train(module_path)
    module = GestureModule.load(module_path)

    # 预测
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([100, 100])])
    root = './datas'
    test_ds = GestureDataset(root, train=False, transform=trans)

    test_loader = DataLoader(test_ds, shuffle=True)
    t_cnt = len(test_loader)
    cnt = 0
    for x, y in test_loader:
        pred_y = module(x)
        print('真实值：', y[0], '预测值：', pred_y[0].argmax().item())
        if int(pred_y[0].argmax().item()) == int(y[0]):
            cnt += 1
    print('正确率：', cnt / t_cnt)
