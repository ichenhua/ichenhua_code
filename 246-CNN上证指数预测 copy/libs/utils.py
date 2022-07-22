import baostock as bs
import pandas as pd
import os
import numpy as np


# 获取大盘数据
def kdata_df():
    file_path = './kline_data.csv'
    if not os.path.exists(file_path):
        lg = bs.login()
        rs = bs.query_history_k_data_plus("sh.000001",
                                          "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                          start_date='2018-01-01',
                                          end_date='2023-03-31',
                                          frequency="d")

        bs.logout()
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['pred_close'] = df['close'].shift(-1)
        df['pred_pctChg'] = df['pctChg'].shift(-1)

        # 结果集输出到csv文件
        df.to_csv(file_path, index=False)

    return pd.read_csv(file_path)


# 格式化数据
def kdata_list():
    df = kdata_df()
    lst = []
    for index in range(59, len(df)-5):
        row = df['pctChg'][index - 59:index + 1].tolist()
        y = (df.loc[index+5, 'close'] - df.loc[index, 'close'])/df.loc[index, 'close']
        row.append(y * 100)
        lst.append(row)
    return np.array(lst)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

lr = LinearRegression()

kdata = kdata_list()

x = kdata[:, :-1]
y = kdata[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y)

lr.fit(x_train, y_train)

print(lr.score(x_test, y_test))

y_pred = lr.predict(x_test)
x_bar = np.arange(len(y_pred))
plt.plot(x_bar, y_pred, label='y_pred')
plt.plot(x_bar, y_test, label='y_test')
plt.legend()
plt.show()




