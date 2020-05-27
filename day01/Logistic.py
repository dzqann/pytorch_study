#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 19:44 
# @Author : DZQ
# @File : Logistic.py

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


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        return self.sm(x)


logistic_model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
data = get_data()
x_data = Tensor([x[:-1] for x in data])
y_data = Tensor([x[-1] for x in data])
for epoch in range(50000):
    x = Variable(x_data)
    y = Variable(y_data).unsqueeze(1)
    out = logistic_model(x)
    loss = criterion(out, y)
    print_loss = loss.data

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mask = out.ge(0.5).float()
    acc = (mask == y).sum().data / y.shape[0]
    if (epoch + 1) % 1000 == 0:
        print("*" * 20)
        print("epoch {}".format(epoch + 1))
        print("loss is {:.4f}".format(print_loss))
        print("acc: {:.4f}".format(acc.detach().numpy()))
        print("*" * 20)
print("weight:{}".format(logistic_model.lr.weight))
w0, w1 = logistic_model.lr.weight[0]
w0 = w0.data
w1 = w1.data
b = logistic_model.lr.bias.data
plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
draw()
plt.show()
