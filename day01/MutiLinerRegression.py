#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 18:00 
# @Author : DZQ
# @File : MutiLinerRegression.py

# %%

import torch
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.autograd import Variable
import numpy as np


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(w_target) + b_target[0]


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        return self.poly(x)


model = poly_model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoach = 0
while True:
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoach + 1
    if print_loss < 1e-3:
        break

print("Loss: {:.6f} after {} batches".format(loss.data, epoach))

parameters = list()
for each in model.parameters():
    parameters.append(each)
w_result = parameters[0]
b_result = parameters[1]
w_result = w_result.detach().numpy()[0]
b_result = b_result.detach().numpy()
print(w_result)
print(b_result)
print("==> Learned function:   y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3".format(b_result[0], w_result[0],
                                                                                       w_result[1],
                                                                                       w_result[2]))

batch_x, batch_y = get_batch()
model.eval()
predict = model(batch_x)
predict = predict.data.numpy()
batch_x = batch_x.numpy()
x_test = [x[0] for x in batch_x]
print(x_test)
y_test = batch_y.numpy()
x = np.linspace(min(x_test), max(x_test), 256, endpoint=True)  # 获取x坐标
y = b_result[0] + w_result[0] * x + w_result[1] * (x ** 2) + w_result[2] * (x ** 3)
plt.scatter(x_test, y_test, label='real curve', color='b')
plt.plot(x, y, label="predict", color="r")
plt.legend()
plt.show()
