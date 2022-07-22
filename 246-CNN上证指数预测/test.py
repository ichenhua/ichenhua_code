#模型测试
import torch
import torch.nn as nn
import torch.utils.data as data

from libs.utils import SModule, SDataset

loss_fn = nn.MSELoss()
module = SModule()

module = torch.load('./m.pkl')
test_ds = SDataset(train=False)
test_loader = data.DataLoader(test_ds, batch_size=10)

for x,y in test_loader:
    y_hat = module(x)
    loss = loss_fn(y_hat, y)
    for y1, y2 in zip(y, y_hat):
        print('真实值:', int(y1.item()), '预测值:', int(y2.item()))
    print('loss:', loss.item())