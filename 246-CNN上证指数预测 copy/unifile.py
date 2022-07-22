# Pytorch实现CNN网络预测上证指数

# 对股民来说，上证指数的涨跌关系着整个市场情绪，在大盘上涨的大前提下操作个股，风险才可控，
# 所以对上证指数的涨跌进行预测，是很有意义的。本文介绍用Pytorch实现CNN网络，来预测上证指数完整过程。

# 1、获取大盘数据
# 都说tushare包，但是获取大盘数据需要积分等级，收费就收费，免费就免费，搞个积分门槛，很讨厌。
# 所以使用替代方案：baostock，文档：http://baostock.com/baostock/index.php/%E6%8C%87%E6%95%B0%E6%95%B0%E6%8D%AE

import baostock as bs
import pandas as pd
import os

def kdata_df():
    file_path = './kline_data.csv'
    if not os.path.exists(file_path):
        lg = bs.login()
        rs = bs.query_history_k_data_plus("sh.000001",
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date='2018-01-01',
            end_date='2022-03-29',
            frequency="d")

        bs.logout()
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['pred_close'] = df['close'].shift(-1)
        
        # 结果集输出到csv文件
        df.to_csv(file_path, index=False)

    return pd.read_csv(file_path)



# 2、格式化数据
# 首先，按天为单位，获取前60个交易日的价格、成交量数据，构建矩阵作为输入数据
# 然后，将后一天的收盘价，作为预测价格输出值

def kdata_list():
    df = kdata_df()
    lst = []
    for index in range(59, len(df)):
        x_col = ['open', 'high', 'low', 'close', 'volume']
        x_val = df.loc[index - 59:index, x_col].values.astype('float32')
        x_val = x_val.reshape(1, x_val.shape[0], x_val.shape[1])
        y_val = df.loc[index, 'pred_close'].astype('float32')
        y_val = y_val.reshape(1)
        lst.append((x_val, y_val))
    return lst




# 3、切分训练集和测试集

import torch.utils.data as data

class SDataset(data.Dataset):
    def __init__(self, train=True):
        # 获取转化后的数据
        lst = kdata_list()
        train_num = int(len(lst)*0.9)
        if train:
            self.kdata = lst[:train_num]
        else:
            self.kdata = lst[train_num:-1]

    def __len__(self):
        return len(self.kdata)

    def __getitem__(self, index):
        x = self.kdata[index][0]
        y = self.kdata[index][1]
        return x, y


# from torchvision import transforms
# train_ds = SDataset(transform=transforms.ToTensor())
# train_loader = data.DataLoader(train_ds, batch_size=5, shuffle=True)
# for x,y in train_loader:
#     print(x.shape)
#     exit()



# 4、定义CNN模型

import torch.nn as nn

class SModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),  #(1, 5, 60)
            nn.BatchNorm2d(32),
            nn.ReLU()  #(32, 5, 60)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  #(32, 5, 60)
            nn.BatchNorm2d(64),
            nn.ReLU()  #(64, 5, 60)
        )
        self.out = nn.Linear(64 * 5 * 60, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        return x


module = SModule()
# print(module)



# 5、模型训练

import torch

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(module.parameters(), lr=0.05)

train_ds = SDataset()
train_loader = data.DataLoader(train_ds, batch_size=100, shuffle=True)

for epoch in range(1000):
    min_loss = 1000
    for x, y in train_loader:
        y_hat = module(x)
        loss = loss_fn(y_hat, y)

        print('epoch:', epoch, 'loss:', loss.item())
        if float(loss.item()) < min_loss:
            torch.save(module, './m.pkl')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()







# 7、大盘预测

import torch

# module = torch.load('./m.pkl')

lst = kdata_list()
x = torch.tensor([lst[-1][0]])
print(x.size())
exit()
y_hat = module(x)

print('预测值:', y_hat.item())

exit()

