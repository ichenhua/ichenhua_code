import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 32, 5, 1, 2), 
    nn.BatchNorm2d(32),
    nn.ReLU(), 
    nn.MaxPool2d(2)
)

print(model)

x = torch.randn([100, 1, 60, 5])

for i in range(len(model)):
    x = model[i](x)
    print(x.size())