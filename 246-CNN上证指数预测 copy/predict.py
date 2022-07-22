# 大盘预测

import torch

from libs.utils import kdata_list
from libs.utils import SModule

module = torch.load('./m.pkl')

lst = kdata_list()
x = torch.tensor([lst[-1][0]])
y_hat = module(x)

print('预测值:', y_hat.item())

