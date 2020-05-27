#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/26 22:37 
# @Author : DZQ
# @File : MutiRegression.py

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

w = [1.5, 1.2, -2.5, -13.1, 4]

x = torch.unsqueeze(torch.linspace(-1, 3, 300), 1)
y = 1.5 + x * 1.2 - x.pow(2) * 1.5 - x.pow(3) * 13.1 + x.pow(4) * 4 + 0.2 * torch.rand(x.size())
# x = torch.cat([x ** i for i in range(1, 5)], 1)


# plt.plot(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(4, 1)

    def get_y(self, x):
        return torch.cat([x ** i for i in range(1, 5)], 1)

    def forward(self, x):
        x = self.get_y(x)
        return self.layer(x)


net = Net()
optimiter = optim.SGD(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()
plt.ion()
plt.show()
for i in range(10000):
    output = net(x)
    # print(output)
    loss = loss_func(output, y)
    optimiter.zero_grad()
    loss.backward()
    optimiter.step()
    # print(loss.data)
    if i % 200 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
        plt.pause(0.1)
plt.show()