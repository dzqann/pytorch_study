#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/26 21:55 
# @Author : DZQ
# @File : Regression.py

import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
from matplotlib import pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2 * x + 3 + torch.rand(x.size()) * 0.15


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()

plt.ion()
plt.show()
loss_list = list()
for i in range(1000):
    output = net(x)
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data.detach().numpy())
    if i % 10 == 0:
        plt.cla()
        plt.plot(range(len(loss_list)), loss_list)
        plt.pause(0.1)
