#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 22:08 
# @Author : DZQ
# @File : MyMutiLineRegression.py

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def union(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


def f(x):
    return x.mm(w_target) + b_target


def get_batch(batch_size=32):
    x_data = torch.randn(batch_size)
    x_train = union(x_data)
    y_train = f(x_train)
    return Variable(x_train), Variable(y_train)


class MutiLineRegression(nn.Module):
    def __init__(self):
        super(MutiLineRegression, self).__init__()
        self.lr = nn.Linear(3, 1)

    def forward(self, x):
        return self.lr(x)


model = MutiLineRegression()
optimster = optim.SGD(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

steps = 0
while True:
    x_data, y_data = get_batch()
    out = model(x_data)
    loss = crit(out, y_data)
    if loss.data < 1e-8:
        break
    optimster.zero_grad()
    loss.backward()
    optimster.step()
    steps += 1
    if steps % 100 == 0:
        print("step: {}, loss: {:.8f}".format(steps, loss.data))

model.eval()
print(model.lr.weight[0].detach().numpy())
print(model.lr.bias.detach().numpy())
bias = model.lr.bias.detach().numpy()
weight = model.lr.weight[0].detach().numpy()
x_data, y_data = get_batch()
x_data = x_data.numpy()
x_data = x_data[:,0]
y_data = y_data.detach().numpy()
plt.plot(x_data, y_data, 'ro')
line = np.linspace(x_data.min(), x_data.max(), 500)
predict = bias + weight[0] * line + weight[1] * (line ** 2) + weight[2] * (line ** 3)
plt.plot(line, predict, 'b')
plt.show()
