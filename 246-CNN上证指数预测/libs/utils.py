import baostock as bs
import pandas as pd
import os

# 获取大盘数据
def kdata_df():
    file_path = './kline_data.csv'
    if not os.path.exists(file_path):
        lg = bs.login()
        rs = bs.query_history_k_data_plus("sh.000001",
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date='2018-01-01',
            end_date='2022-03-31',
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


# 格式化数据
def kdata_list():
    df = kdata_df()
    lst = []
    for index in range(59, len(df)):
        x_col = ['open', 'high', 'low', 'close', 'volume', 'amount']
        x_val = df.loc[index - 59:index, x_col].values.astype('float32')
        x_val = x_val.reshape(1, x_val.shape[0], x_val.shape[1])
        y_val = df.loc[index, 'pred_close'].astype('float32')
        y_val = y_val.reshape(1)
        lst.append((x_val, y_val))
    return lst


# 切分训练集和测试集
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


# 定义CNN模型
import torch.nn as nn

class SModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  #(1, 6, 60)
            nn.BatchNorm2d(32),
            nn.ReLU()  #(32, 6, 60)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  #(32, 6, 60)
            nn.BatchNorm2d(64),
            nn.ReLU()  #(64, 6, 60)
        )
        self.out = nn.Linear(64 * 6 * 60, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        return x



        