#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 22:40 
# @Author : DZQ
# @File : MyLogistic.py

from torch import nn
from torch.autograd import Variable
from torch import optim
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    with open("./dataset.txt", "r", encoding="utf8") as file:
        data_list = file.readlines()
        data_list = [x.strip() for x in data_list]
        data_list = [x.split(',') for x in data_list]
        data = [(float(x[0].strip()), float(x[1].strip()), int(x[2].strip())) for x in data_list]
        file.close()
    return data


def draw():
    data = get_data()
    x0 = list(filter(lambda x: x[-1] == 0, data))
    x1 = list(filter(lambda x: x[-1] == 1, data))
    plot_x0_0 = [x[0] for x in x0]
    plot_x0_1 = [x[1] for x in x0]
    plot_x1_0 = [x[0] for x in x1]
    plot_x1_1 = [x[1] for x in x1]
    plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x0')
    plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x1')


class MyLogistic(nn.Module):
    def __init__(self):
        super(MyLogistic, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        return self.sm(self.lr(x))


data = get_data()
x_data = Tensor([x[:-1] for x in data])
y_data = Tensor([x[-1] for x in data])
x = Variable(x_data)
y = Variable(y_data).unsqueeze(1)
module = MyLogistic()
optimister = optim.SGD(module.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.BCELoss()
loss_list = list()
for i in range(50000):
    out = module(x)
    loss = criterion(out, y)
    loss_data = loss.data
    loss_list.append(loss.data)
    optimister.zero_grad()
    loss.backward()
    optimister.step()
    if (i + 1) % 1000 == 0:
        print("*" * 30)
        print("epoch: {}".format(i + 1))
        print("loss: {:.5f}".format(loss_data))
        print("*" * 30)

draw()
weight = module.lr.weight.detach().numpy()
w0 = weight[0]
bias = module.lr.bias.data

w0, w1 = module.lr.weight[0]
w0 = w0.data
w1 = w1.data
b = module.lr.bias.data
plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
draw()
plt.show()
plt.plot(range(50000), loss_list)
plt.show()